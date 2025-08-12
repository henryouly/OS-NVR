// SPDX-License-Identifier: GPL-2.0-or-later

package openvino

import (
	"nvr/pkg/monitor"
	"time"
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

func parseConfig(c monitor.Config) (*config, bool, error) { //nolint:funlen
	enable := true
	return &config{
		monitorID:       c.ID(),
		hwaccel:         "auto",
		ffmpegLogLevel:  "info",
		timestampOffset: 0,
		thresholds:      make(thresholds),
		cropX:           0.0,
		cropY:           0.0,
		cropSize:        100.0,
		minSize:        64,
		maxSize:        64,
		detectorName:   "yolo",
		grayMode:       false,
		feedRate:       5.0,
		recDuration:    60 * time.Second,
		useSubStream:   false,
	}, enable, nil
}

const (

)

func (c *config) fillMissing() {

}

// The WebUI shouldn't allow the user to save invalid values, this is more of
// a sanity check in case of failed migration or manual config file edits.
func (c *config) validate() error {
	return nil
}
