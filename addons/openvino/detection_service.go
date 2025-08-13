package openvino

import (
	"image"
	"log"
	"sync"
)

type DetectionTask struct {
  StreamName string
  Image      image.Image
}

type DetectionService struct {
	queue         chan DetectionTask
	wg            sync.WaitGroup
	objectDetector *ObjectDetector
}

func NewDetectionService(objectDetector *ObjectDetector) *DetectionService {
	return &DetectionService{
		queue:         make(chan DetectionTask, 256),
		objectDetector: objectDetector,
	}
}

func (ds *DetectionService) Start(numWorkers int) {
  log.Printf("Starting %d shared detection workers", numWorkers)
	for i := 0; i < numWorkers; i++ {
		ds.wg.Add(1)
		go ds.worker(i + 1)
	}
}

func (ds *DetectionService) Stop() {
	close(ds.queue)
	ds.wg.Wait()
}

func (ds *DetectionService) worker(id int) {
	defer ds.wg.Done()

	log.Printf("Detection worker #%d started", id)

	// This loop will automatically exit when the 'queue' channel is closed.
	for task := range ds.queue {
		detection, err := ds.objectDetector.DetectObjects(task.Image)
    if err != nil {
      log.Printf("[Worker %d] Failed to run detection for stream %s: %v", id, task.StreamName, err)
    } else {
			if len(detection) > 0 && detection[0].ClassId == 0 {
				log.Printf("[Worker %d] Successfully processed frame from stream %s: %v", id, task.StreamName, detection)
			}
    }
  }
  log.Printf("Detection worker #%d shut down", id)
}

func (ds *DetectionService) AddTask(task DetectionTask) {
	ds.queue <- task
}
