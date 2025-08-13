// SPDX-License-Identifier: GPL-2.0-or-later

package openvino

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"nvr/pkg/ffmpeg"
	"nvr/pkg/monitor"
	"strconv"
	"time"

	"gopkg.in/yaml.v3"
)

var (
	openvinoConfig struct {
		Host      string `yaml:"host"`
		ModelName string `yaml:"model_name"`
		NumThreads int    `yaml:"num_threads"`
		ModelConfig struct {
			InputTensor            string  `yaml:"input_tensor"`
			ConfidenceThreshold   float64 `yaml:"confidence_threshold"`
			IOUThreshold          float64 `yaml:"iou_threshold"`
			NumClasses            int      `yaml:"num_classes"`
			ClassNames            []string `yaml:"class_names"`
			Preprocess            struct {
				Resize struct {
					Width  int `yaml:"width"`
					Height int `yaml:"height"`
				} `yaml:"resize"`
				Normalize struct {
					Mean   []float64 `yaml:"mean"`
				} `yaml:"normalize"`
			} `yaml:"preprocess"`
		} `yaml:"model_config"`
	}
)

type config struct {
	monitorID       string
	hwaccel         string
	ffmpegLogLevel  string
	timestampOffset time.Duration
	thresholds      thresholds
	cropX           float64
	cropY           float64
	cropSize        float64
	// mask            mask
	minSize         float64
	maxSize         float64
	detectorName    string
	grayMode        bool
	feedRate        float64
	recDuration     time.Duration
	useSubStream    bool
}

type mask struct {
	Enable bool           `json:"enable"`
	Area   ffmpeg.Polygon `json:"area"`
}

func parseConfig(c monitor.Config) (*config, bool, error) { //nolint:funlen
	enable := c.Get("enable") == "true"
	if !enable {
		return nil, false, nil
	}

	return &config{
		monitorID:       c.ID(),
		hwaccel:         c.Hwaccel(),
		ffmpegLogLevel:  c.LogLevel(),
		timestampOffset: 0,
		thresholds:      make(thresholds),
		cropX:           0,
		cropY:           0,
		cropSize:        100,
		// mask:            mask,
		minSize:         64,
		maxSize:         64,
		detectorName:    "yolov8n",
		grayMode:        false,
		feedRate:        5.0,
		recDuration:     120 * time.Second,
		useSubStream:    false,
	}, enable, nil
}

func parseRawConfig(path string) error {
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read config file: %w", err)
	}

	if err := yaml.Unmarshal(data, &openvinoConfig); err != nil {
		return fmt.Errorf("unmarshal config: %w", err)
	}

	return nil
}

func parseThresholds(rawThresholds string) (thresholds, error) {
	if rawThresholds == "" {
		return nil, nil
	}

	var t thresholds
	err := json.Unmarshal([]byte(rawThresholds), &t)
	if err != nil {
		return nil, err
	}
	for key, thresh := range t {
		if thresh == -1 {
			delete(t, key)
		}
	}
	return t, nil
}

func parseDuration(rawDuration string) (time.Duration, error) {
	if rawDuration == "" {
		return 0, nil
	}
	recDurationFloat, err := strconv.ParseFloat(rawDuration, 64)
	if err != nil {
		return 0, fmt.Errorf("parse duration: %w", err)
	}
	return time.Duration(recDurationFloat * float64(time.Second)), nil
}

const (
	defaultCropSize    = 100
	defaultFeedRate    = 0.2
	defaultRecDuration = 120 * time.Second
)

func (c *config) fillMissing() {
	if c.thresholds == nil {
		c.thresholds = thresholds{}
	}
	if c.cropSize == 0 {
		c.cropSize = defaultCropSize
	}
	if c.feedRate == 0 {
		c.feedRate = defaultFeedRate
	}
	if c.recDuration == 0 {
		c.recDuration = defaultRecDuration
	}
}

// Validate errors.
var (
	ErrInvalidCropSize = errors.New("invalid crop size")
	ErrInvalidCropX    = errors.New("invalid cropX")
	ErrInvalidCropY    = errors.New("invalid cropY")
	ErrInvalidFeedRate = errors.New("invalid feed rate")
	ErrInvalidDuration = errors.New("invalid duration")
)

// The WebUI shouldn't allow the user to save invalid values, this is more of
// a sanity check in case of failed migration or manual config file edits.
func (c *config) validate() error {
	if c.cropSize < 0 || c.cropSize > 100 {
		return fmt.Errorf("%w: %v", ErrInvalidCropSize, c.cropSize)
	}
	if c.cropX < 0 || c.cropX > 100 {
		return fmt.Errorf("%w: %v", ErrInvalidCropX, c.cropX)
	}
	if c.cropY < 0 || c.cropY > 100 {
		return fmt.Errorf("%w: %v", ErrInvalidCropY, c.cropY)
	}
	if c.feedRate <= 0 {
		return fmt.Errorf("%w: %v", ErrInvalidFeedRate, c.feedRate)
	}
	if c.recDuration < 0 {
		return fmt.Errorf("%w: %v", ErrInvalidDuration, c.recDuration)
	}
	return nil
}
