package main

import (
	"bufio"
	"bytes"
	"crypto/sha256"
	"encoding/base64"
	"encoding/hex"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io"
	"mime/multipart"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"
)

const (
	defaultAPIURL  = "https://apinebula.com"
	defaultModel   = "gpt-image-2"
	defaultSize    = "1024x1024"
	defaultQuality = "auto"
	defaultOutDir  = "output/imagegen"
)

var retryable = map[int]bool{429: true, 500: true, 502: true, 503: true, 504: true, 524: true}

type commonArgs struct {
	model       string
	size        string
	quality     string
	n           int
	outDir      string
	modelSet    bool
	force       bool
	dryRun      bool
	maxAttempts int
	timeout     time.Duration
}

type runtimeConfig struct {
	apiURL string
	apiKey string
	model  string
}

type apiResponse struct {
	Data []struct {
		B64JSON string `json:"b64_json"`
		URL     string `json:"url"`
	} `json:"data"`
}

type batchJob struct {
	Prompt  string `json:"prompt"`
	Model   string `json:"model,omitempty"`
	Size    string `json:"size,omitempty"`
	Quality string `json:"quality,omitempty"`
	N       int    `json:"n,omitempty"`
	Out     string `json:"out,omitempty"`
}

func endpoint(base, operation string) string {
	base = strings.TrimRight(strings.TrimSpace(base), "/")
	if base == "" {
		base = defaultAPIURL
	}
	if !strings.HasSuffix(base, "/v1") {
		base += "/v1"
	}
	return base + "/images/" + operation
}

func validateCommon(args commonArgs) error {
	if args.n < 1 || args.n > 10 {
		return errors.New("n must be from 1 to 10")
	}
	if args.maxAttempts < 1 {
		return errors.New("--max-attempts must be at least 1")
	}
	if args.timeout <= 0 {
		return errors.New("--timeout must be positive")
	}
	if args.quality != "low" && args.quality != "medium" && args.quality != "high" && args.quality != "auto" {
		return errors.New("quality must be low, medium, high, or auto")
	}
	if args.size != "auto" {
		parts := strings.Split(args.size, "x")
		if len(parts) != 2 {
			return errors.New("size must be 'auto' or WIDTHxHEIGHT")
		}
		w, e1 := strconv.Atoi(parts[0])
		h, e2 := strconv.Atoi(parts[1])
		if e1 != nil || e2 != nil || w < 1 || h < 1 {
			return errors.New("size must be 'auto' or WIDTHxHEIGHT")
		}
	}
	return nil
}

func promptValue(prompt, promptFile string) (string, error) {
	if (prompt == "") == (promptFile == "") {
		return "", errors.New("provide exactly one of --prompt or --prompt-file")
	}
	if promptFile != "" {
		data, err := os.ReadFile(promptFile)
		if err != nil {
			return "", fmt.Errorf("could not read prompt file: %w", err)
		}
		prompt = string(data)
	}
	prompt = strings.TrimSpace(prompt)
	if prompt == "" {
		return "", errors.New("prompt must not be empty")
	}
	return prompt, nil
}

func outputPaths(out, outDir, prompt string, n int) []string {
	if out == "" {
		sum := sha256.Sum256([]byte(prompt))
		out = "image-" + hex.EncodeToString(sum[:6]) + ".png"
	}
	if filepath.Dir(out) == "." {
		out = filepath.Join(outDir, out)
	}
	ext := filepath.Ext(out)
	if ext == "" {
		ext = ".png"
		out += ext
	}
	if n == 1 {
		return []string{out}
	}
	base := strings.TrimSuffix(out, ext)
	paths := make([]string, n)
	for i := range paths {
		paths[i] = fmt.Sprintf("%s-%d%s", base, i+1, ext)
	}
	return paths
}

func checkOutputs(paths []string, force bool) error {
	if force {
		return nil
	}
	for _, path := range paths {
		if _, err := os.Stat(path); err == nil {
			return fmt.Errorf("refusing to overwrite existing file: %s", path)
		}
	}
	return nil
}

// codexConfigPath 返回用户级 Codex 配置文件路径。
//
// @param name 配置文件名称。
// @return 默认配置文件路径；无法定位用户目录时返回空字符串。
func codexConfigPath(name string) string {
	// 定位当前用户主目录。
	home, err := os.UserHomeDir()
	if err != nil || strings.TrimSpace(home) == "" {
		return ""
	}

	// 返回 Codex 用户级配置文件。
	return filepath.Join(home, ".codex", name)
}

// stripTOMLComment 移除 TOML 行尾注释。
//
// @param line 原始 TOML 行。
// @return 去除注释后的内容。
func stripTOMLComment(line string) string {
	var quote rune
	escaped := false
	for index, value := range line {
		// 处理字符串内部的转义和闭合。
		if quote != 0 {
			if quote == '"' && escaped {
				escaped = false
				continue
			}
			if quote == '"' && value == '\\' {
				escaped = true
				continue
			}
			if value == quote {
				quote = 0
			}
			continue
		}

		// 进入字符串后忽略其中的 #。
		if value == '"' || value == '\'' {
			quote = value
			continue
		}

		// 截断真正的注释内容。
		if value == '#' {
			return strings.TrimSpace(line[:index])
		}
	}
	return strings.TrimSpace(line)
}

// parseTOMLString 解析本工具支持的简单 TOML 字符串值。
//
// @param value TOML 等号右侧的原始值。
// @return 解析后的字符串和值是否有效。
func parseTOMLString(value string) (string, bool) {
	// 清理值两侧空白。
	value = strings.TrimSpace(value)
	if value == "" {
		return "", false
	}

	// 解析双引号字符串，支持标准转义。
	if strings.HasPrefix(value, "\"") {
		parsed, err := strconv.Unquote(value)
		if err != nil {
			return "", false
		}
		return strings.TrimSpace(parsed), true
	}

	// 解析单引号字符串。
	if strings.HasPrefix(value, "'") && strings.HasSuffix(value, "'") && len(value) >= 2 {
		return strings.TrimSpace(value[1 : len(value)-1]), true
	}

	// 兼容无引号字符串。
	return strings.TrimSpace(value), true
}

// readAuthFile 从 Codex auth.json 读取 API 密钥。
//
// @param path auth.json 文件路径。
// @return 文件中的 API 配置和读取错误。
func readAuthFile(path string) (runtimeConfig, error) {
	var config runtimeConfig
	if strings.TrimSpace(path) == "" {
		return config, nil
	}

	// 读取认证文件，不存在时视为未配置。
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return config, nil
	}
	if err != nil {
		return config, fmt.Errorf("could not read auth file: %w", err)
	}

	// 解析 Codex 标准密钥字段。
	var auth struct {
		OpenAIAPIKey string `json:"OPENAI_API_KEY"`
	}
	if err := json.Unmarshal(data, &auth); err != nil {
		return config, fmt.Errorf("could not parse auth file: %w", err)
	}
	config.apiKey = strings.TrimSpace(auth.OpenAIAPIKey)
	return config, nil
}

// readCodexConfig 从 Codex config.toml 读取供应商地址和模型。
//
// @param path config.toml 文件路径。
// @return 文件中的 API 配置和读取错误。
func readCodexConfig(path string) (runtimeConfig, error) {
	var config runtimeConfig
	if strings.TrimSpace(path) == "" {
		return config, nil
	}

	// 读取配置文件，不存在时视为未配置。
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return config, nil
	}
	if err != nil {
		return config, fmt.Errorf("could not read config file: %w", err)
	}

	// 扫描 Codex 选定模型和供应商配置。
	scanner := bufio.NewScanner(bytes.NewReader(data))
	currentTable := ""
	modelProvider := ""
	baseURL := ""
	providerBaseURLs := map[string]string{}
	for scanner.Scan() {
		line := stripTOMLComment(scanner.Text())
		if line == "" {
			continue
		}
		if strings.HasPrefix(line, "[") && strings.HasSuffix(line, "]") {
			currentTable = strings.TrimSpace(line[1 : len(line)-1])
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) != 2 {
			continue
		}
		key := strings.Trim(strings.TrimSpace(parts[0]), "\"'")
		value, ok := parseTOMLString(parts[1])
		if !ok {
			continue
		}
		if currentTable == "" {
			switch key {
			case "model_provider":
				modelProvider = value
			case "model":
				config.model = value
			case "base_url":
				baseURL = value
			}
			continue
		}
		if strings.HasPrefix(currentTable, "model_providers.") && key == "base_url" {
			provider := strings.TrimPrefix(currentTable, "model_providers.")
			providerBaseURLs[provider] = value
		}
	}
	if err := scanner.Err(); err != nil {
		return config, fmt.Errorf("could not scan config file: %w", err)
	}

	if selected := strings.TrimSpace(modelProvider); selected != "" {
		if resolved := strings.TrimSpace(providerBaseURLs[selected]); resolved != "" {
			config.apiURL = resolved
		}
	}
	if strings.TrimSpace(config.apiURL) == "" {
		config.apiURL = strings.TrimSpace(baseURL)
	}
	return config, nil
}

// loadRuntimeConfig 合并 Codex 配置文件和环境变量中的运行时配置。
//
// @return 合并后的运行时配置和读取错误。
func loadRuntimeConfig() (runtimeConfig, error) {
	// 先读取 Codex 认证文件。
	config, err := readAuthFile(codexConfigPath("auth.json"))
	if err != nil {
		return config, err
	}

	// 再读取 Codex 主配置文件。
	codexConfig, err := readCodexConfig(codexConfigPath("config.toml"))
	if err != nil {
		return config, err
	}
	config.apiURL = codexConfig.apiURL
	config.model = codexConfig.model

	// 使用环境变量覆盖 Codex 配置，保持已有行为和临时覆盖能力。
	if apiURL := strings.TrimSpace(os.Getenv("CODEX_API_URL")); apiURL != "" {
		config.apiURL = apiURL
	}
	if apiKey := strings.TrimSpace(os.Getenv("CODEX_API_KEY")); apiKey != "" {
		config.apiKey = apiKey
	}
	return config, nil
}

// apiBase 返回实际请求使用的 API 基础地址。
//
// @param config 已合并的运行时配置。
// @return API 基础地址。
func apiBase(config runtimeConfig) string {
	// 配置缺失时使用默认中转地址。
	if strings.TrimSpace(config.apiURL) == "" {
		return defaultAPIURL
	}
	return config.apiURL
}

// apiKey 返回实际请求使用的 API 密钥。
//
// @param config 已合并的运行时配置。
// @return API 密钥和校验错误。
func apiKey(config runtimeConfig) (string, error) {
	// 校验密钥是否已通过环境变量或 auth.json 配置。
	key := strings.TrimSpace(config.apiKey)
	if key == "" {
		return "", errors.New("CODEX_API_KEY or ~/.codex/auth.json OPENAI_API_KEY is not set; set it locally, then retry")
	}
	return key, nil
}

func doRequest(client *http.Client, makeRequest func() (*http.Request, error), maxAttempts int) ([]byte, error) {
	var last string
	for attempt := 1; attempt <= maxAttempts; attempt++ {
		req, err := makeRequest()
		if err != nil {
			return nil, err
		}
		resp, err := client.Do(req)
		if err == nil {
			body, readErr := io.ReadAll(resp.Body)
			resp.Body.Close()
			if readErr != nil {
				return nil, errors.New("could not read API response")
			}
			if resp.StatusCode >= 200 && resp.StatusCode < 300 {
				return body, nil
			}
			last = fmt.Sprintf("API request failed with HTTP %d", resp.StatusCode)
			if !retryable[resp.StatusCode] || attempt == maxAttempts {
				return nil, errors.New(last)
			}
		} else {
			last = "API request failed: network error or timeout"
			if attempt == maxAttempts {
				return nil, errors.New(last)
			}
		}
		time.Sleep(time.Duration(750*(1<<(attempt-1))) * time.Millisecond)
	}
	return nil, errors.New(last)
}

func decodeResponse(raw []byte, client *http.Client) ([][]byte, error) {
	var result apiResponse
	if err := json.Unmarshal(raw, &result); err != nil {
		return nil, errors.New("API returned invalid JSON")
	}
	if len(result.Data) == 0 {
		return nil, errors.New("API response contains no image data")
	}
	images := make([][]byte, 0, len(result.Data))
	for _, item := range result.Data {
		if item.B64JSON != "" {
			data, err := base64.StdEncoding.DecodeString(item.B64JSON)
			if err != nil {
				return nil, errors.New("API returned invalid base64 image data")
			}
			images = append(images, data)
		} else if item.URL != "" {
			if _, err := url.ParseRequestURI(item.URL); err != nil {
				return nil, errors.New("API returned an invalid image URL")
			}
			resp, err := client.Get(item.URL)
			if err != nil || resp.StatusCode < 200 || resp.StatusCode >= 300 {
				if resp != nil {
					resp.Body.Close()
				}
				return nil, errors.New("could not download image URL")
			}
			data, err := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err != nil {
				return nil, errors.New("could not read downloaded image")
			}
			images = append(images, data)
		} else {
			return nil, errors.New("API image entry contains neither b64_json nor url")
		}
	}
	return images, nil
}

func saveImages(images [][]byte, paths []string) ([]string, error) {
	if len(images) != len(paths) {
		return nil, fmt.Errorf("API returned %d image(s), expected %d", len(images), len(paths))
	}
	abs := make([]string, len(paths))
	for i, path := range paths {
		if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
			return nil, fmt.Errorf("could not create output directory: %w", err)
		}
		if err := os.WriteFile(path, images[i], 0644); err != nil {
			return nil, fmt.Errorf("could not write output: %w", err)
		}
		abs[i], _ = filepath.Abs(path)
	}
	return abs, nil
}

// generate 调用图片生成接口并保存返回图片。
//
// @param prompt 图片生成提示词。
// @param out 输出文件路径。
// @param args 通用命令参数。
// @return 生成结果摘要和执行错误。
func generate(prompt, out string, args commonArgs) (map[string]any, error) {
	// 加载运行时配置。
	config, err := loadRuntimeConfig()
	if err != nil {
		return nil, err
	}

	// 校验通用参数。
	if err := validateCommon(args); err != nil {
		return nil, err
	}
	// 使用 Codex 配置中的模型作为默认值。
	if strings.TrimSpace(config.model) != "" && !args.modelSet {
		args.model = config.model
	}
	// 计算并检查输出路径。
	paths := outputPaths(out, args.outDir, prompt, args.n)
	if err := checkOutputs(paths, args.force); err != nil {
		return nil, err
	}
	ep := endpoint(apiBase(config), "generations")
	payload := map[string]any{"model": args.model, "prompt": prompt, "size": args.size, "quality": args.quality, "n": args.n}
	if args.dryRun {
		return map[string]any{"dry_run": true, "endpoint": ep, "payload": payload, "outputs": paths}, nil
	}
	// 读取密钥并发起请求。
	key, err := apiKey(config)
	if err != nil {
		return nil, err
	}
	body, _ := json.Marshal(payload)
	client := &http.Client{Timeout: args.timeout}
	raw, err := doRequest(client, func() (*http.Request, error) {
		req, err := http.NewRequest(http.MethodPost, ep, bytes.NewReader(body))
		if err == nil {
			req.Header.Set("Authorization", "Bearer "+key)
			req.Header.Set("Content-Type", "application/json")
		}
		return req, err
	}, args.maxAttempts)
	if err != nil {
		return nil, err
	}
	images, err := decodeResponse(raw, client)
	if err != nil {
		return nil, err
	}
	outputs, err := saveImages(images, paths)
	if err != nil {
		return nil, err
	}
	return map[string]any{"model": args.model, "size": args.size, "quality": args.quality, "outputs": outputs}, nil
}

// edit 调用图片编辑接口并保存返回图片。
//
// @param prompt 图片编辑提示词。
// @param imagePaths 输入图片路径列表。
// @param mask 可选 PNG 蒙版路径。
// @param out 输出文件路径。
// @param args 通用命令参数。
// @return 编辑结果摘要和执行错误。
func edit(prompt string, imagePaths []string, mask, out string, args commonArgs) (map[string]any, error) {
	// 加载运行时配置。
	config, err := loadRuntimeConfig()
	if err != nil {
		return nil, err
	}

	// 校验通用参数和输入文件。
	if err := validateCommon(args); err != nil {
		return nil, err
	}
	// 使用 Codex 配置中的模型作为默认值。
	if strings.TrimSpace(config.model) != "" && !args.modelSet {
		args.model = config.model
	}
	for _, path := range append(append([]string{}, imagePaths...), mask) {
		if path == "" {
			continue
		}
		if info, err := os.Stat(path); err != nil || info.IsDir() {
			return nil, fmt.Errorf("input file not found: %s", path)
		}
	}
	paths := outputPaths(out, args.outDir, prompt, args.n)
	if err := checkOutputs(paths, args.force); err != nil {
		return nil, err
	}
	ep := endpoint(apiBase(config), "edits")
	fields := map[string]string{"model": args.model, "prompt": prompt, "size": args.size, "quality": args.quality, "n": strconv.Itoa(args.n)}
	if args.dryRun {
		return map[string]any{"dry_run": true, "endpoint": ep, "fields": fields, "images": imagePaths, "mask": mask, "outputs": paths}, nil
	}
	// 读取密钥并组装 multipart 请求。
	key, err := apiKey(config)
	if err != nil {
		return nil, err
	}
	var body bytes.Buffer
	writer := multipart.NewWriter(&body)
	for name, value := range fields {
		_ = writer.WriteField(name, value)
	}
	for _, path := range imagePaths {
		part, err := writer.CreateFormFile("image", filepath.Base(path))
		if err != nil {
			return nil, err
		}
		file, err := os.Open(path)
		if err != nil {
			return nil, err
		}
		_, err = io.Copy(part, file)
		file.Close()
		if err != nil {
			return nil, err
		}
	}
	if mask != "" {
		part, err := writer.CreateFormFile("mask", filepath.Base(mask))
		if err != nil {
			return nil, err
		}
		file, err := os.Open(mask)
		if err != nil {
			return nil, err
		}
		_, err = io.Copy(part, file)
		file.Close()
		if err != nil {
			return nil, err
		}
	}
	writer.Close()
	contentType := writer.FormDataContentType()
	data := append([]byte(nil), body.Bytes()...)
	client := &http.Client{Timeout: args.timeout}
	raw, err := doRequest(client, func() (*http.Request, error) {
		req, err := http.NewRequest(http.MethodPost, ep, bytes.NewReader(data))
		if err == nil {
			req.Header.Set("Authorization", "Bearer "+key)
			req.Header.Set("Content-Type", contentType)
		}
		return req, err
	}, args.maxAttempts)
	if err != nil {
		return nil, err
	}
	images, err := decodeResponse(raw, client)
	if err != nil {
		return nil, err
	}
	outputs, err := saveImages(images, paths)
	if err != nil {
		return nil, err
	}
	return map[string]any{"model": args.model, "size": args.size, "quality": args.quality, "outputs": outputs}, nil
}

func printJSON(value any) {
	encoder := json.NewEncoder(os.Stdout)
	encoder.SetIndent("", "  ")
	encoder.SetEscapeHTML(false)
	_ = encoder.Encode(value)
}

// parseCommon 解析所有命令共享的 CLI 参数。
//
// @param fs 命令专用 FlagSet。
// @param argv 命令参数。
// @param args 通用参数接收对象。
// @return 参数解析错误。
func parseCommon(fs *flag.FlagSet, argv []string, args *commonArgs) error {
	var timeout float64
	fs.StringVar(&args.model, "model", defaultModel, "image model")
	fs.StringVar(&args.size, "size", defaultSize, "auto or WIDTHxHEIGHT")
	fs.StringVar(&args.quality, "quality", defaultQuality, "low, medium, high, or auto")
	fs.IntVar(&args.n, "n", 1, "number of variants (1-10)")
	fs.StringVar(&args.outDir, "out-dir", defaultOutDir, "default output directory")
	fs.BoolVar(&args.force, "force", false, "overwrite existing outputs")
	fs.BoolVar(&args.dryRun, "dry-run", false, "validate without network access")
	fs.IntVar(&args.maxAttempts, "max-attempts", 3, "maximum attempts")
	fs.Float64Var(&timeout, "timeout", 150, "request timeout in seconds")
	if err := fs.Parse(argv); err != nil {
		return err
	}
	// 记录用户是否显式指定模型，避免被 config.toml 的模型覆盖。
	fs.Visit(func(flag *flag.Flag) {
		if flag.Name == "model" {
			args.modelSet = true
		}
	})
	args.timeout = time.Duration(timeout * float64(time.Second))
	return nil
}

type stringList []string

func (s *stringList) String() string { return strings.Join(*s, ",") }
func (s *stringList) Set(value string) error {
	*s = append(*s, value)
	return nil
}

func runGenerate(argv []string) error {
	fs := flag.NewFlagSet("generate", flag.ContinueOnError)
	var prompt, promptFile, out string
	fs.StringVar(&prompt, "prompt", "", "prompt text")
	fs.StringVar(&promptFile, "prompt-file", "", "UTF-8 prompt file")
	fs.StringVar(&out, "out", "", "output path")
	var args commonArgs
	if err := parseCommon(fs, argv, &args); err != nil {
		return err
	}
	finalPrompt, err := promptValue(prompt, promptFile)
	if err != nil {
		return err
	}
	result, err := generate(finalPrompt, out, args)
	if err == nil {
		printJSON(result)
	}
	return err
}

func runEdit(argv []string) error {
	fs := flag.NewFlagSet("edit", flag.ContinueOnError)
	var images stringList
	var prompt, promptFile, mask, out string
	fs.Var(&images, "image", "input image; repeat for multiple images")
	fs.StringVar(&mask, "mask", "", "optional PNG mask")
	fs.StringVar(&prompt, "prompt", "", "prompt text")
	fs.StringVar(&promptFile, "prompt-file", "", "UTF-8 prompt file")
	fs.StringVar(&out, "out", "", "output path")
	var args commonArgs
	if err := parseCommon(fs, argv, &args); err != nil {
		return err
	}
	if len(images) == 0 {
		return errors.New("provide at least one --image")
	}
	finalPrompt, err := promptValue(prompt, promptFile)
	if err != nil {
		return err
	}
	result, err := edit(finalPrompt, images, mask, out, args)
	if err == nil {
		printJSON(result)
	}
	return err
}

func runBatch(argv []string) error {
	fs := flag.NewFlagSet("generate-batch", flag.ContinueOnError)
	var input string
	var concurrency int
	var failFast bool
	fs.StringVar(&input, "input", "", "JSONL input path")
	fs.IntVar(&concurrency, "concurrency", 2, "parallel jobs")
	fs.BoolVar(&failFast, "fail-fast", false, "stop scheduling after a failure")
	var args commonArgs
	if err := parseCommon(fs, argv, &args); err != nil {
		return err
	}
	if input == "" || concurrency < 1 {
		return errors.New("--input is required and --concurrency must be at least 1")
	}
	file, err := os.Open(input)
	if err != nil {
		return fmt.Errorf("could not read batch input: %w", err)
	}
	defer file.Close()
	var jobs []batchJob
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		if strings.TrimSpace(scanner.Text()) == "" {
			continue
		}
		var job batchJob
		if err := json.Unmarshal(scanner.Bytes(), &job); err != nil || strings.TrimSpace(job.Prompt) == "" {
			return errors.New("each JSONL line must contain a non-empty prompt")
		}
		jobs = append(jobs, job)
	}
	if err := scanner.Err(); err != nil || len(jobs) == 0 {
		return errors.New("batch input contains no valid jobs")
	}
	type result struct {
		Index int            `json:"index"`
		OK    bool           `json:"ok"`
		Data  map[string]any `json:"data,omitempty"`
		Error string         `json:"error,omitempty"`
	}
	results := make([]result, len(jobs))
	queue := make(chan int)
	var wg sync.WaitGroup
	var failed bool
	var mu sync.Mutex
	for range concurrency {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for index := range queue {
				job := jobs[index]
				jobArgs := args
				if job.Model != "" {
					jobArgs.model = job.Model
				}
				if job.Size != "" {
					jobArgs.size = job.Size
				}
				if job.Quality != "" {
					jobArgs.quality = job.Quality
				}
				if job.N != 0 {
					jobArgs.n = job.N
				}
				data, err := generate(strings.TrimSpace(job.Prompt), job.Out, jobArgs)
				results[index] = result{Index: index + 1, OK: err == nil, Data: data}
				if err != nil {
					results[index].Error = err.Error()
					mu.Lock()
					failed = true
					mu.Unlock()
				}
			}
		}()
	}
	for index := range jobs {
		mu.Lock()
		stop := failFast && failed
		mu.Unlock()
		if stop {
			break
		}
		queue <- index
	}
	close(queue)
	wg.Wait()
	succeeded, failures := 0, 0
	for _, item := range results {
		if item.Index == 0 {
			continue
		}
		if item.OK {
			succeeded++
		} else {
			failures++
		}
	}
	printJSON(map[string]any{"jobs": results, "succeeded": succeeded, "failed": failures})
	if failures > 0 {
		return errors.New("one or more batch jobs failed")
	}
	return nil
}

func usage() {
	fmt.Fprintln(os.Stderr, "Usage: codex-image2 <generate|generate-batch|edit> [options]")
}

func main() {
	if len(os.Args) < 2 {
		usage()
		os.Exit(2)
	}
	var err error
	switch os.Args[1] {
	case "generate":
		err = runGenerate(os.Args[2:])
	case "generate-batch":
		err = runBatch(os.Args[2:])
	case "edit":
		err = runEdit(os.Args[2:])
	case "--help", "-h", "help":
		usage()
		return
	default:
		err = fmt.Errorf("unknown command: %s", os.Args[1])
	}
	if err != nil {
		fmt.Fprintln(os.Stderr, "error:", err)
		os.Exit(2)
	}
}
