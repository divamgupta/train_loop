from queue import Queue
from threading import Thread, Event, Lock
from multiprocessing import Process, Queue as MPQueue, Event as MPEvent, Lock as MPLock, Manager
import time
import random
import requests
import cv2
import numpy as np

def download_image(url, timeout=10):
    """
    Download a single image from URL.
    
    Args:
        url: Image URL
        timeout: Timeout for image download in seconds
        
    Returns:
        tuple: (opencv_image, url) or (None, None) if failed
    """
    try:
        headers = {
            'User-Agent': 'curl/7.68.0'
        }
        response = requests.get(url, timeout=timeout, headers=headers, allow_redirects=True)
        response.raise_for_status()
        
        # Convert to numpy array
        img_array = np.asarray(bytearray(response.content), dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        
        if img is None:
            return None, None
            
        return img, url
        
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 429:
            print(f"Rate limited (429) for {url}, waiting 5 seconds...")
            time.sleep(5)
        else:
            print(f"HTTP error downloading from {url}: {str(e)}")
        return None, None
    except Exception as e:
        print(f"Failed to download image from {url}: {str(e)}")
        return None, None

class OnlineCVImagesGiver:
    """
    A class that continuously downloads random images from URLs and provides them via a queue.
    Call get() to retrieve an OpenCV image and its URL.
    """
    
    def __init__(self, image_urls, timeout=10, num_threads=20, queue_size=100, cache_size=50, use_multiprocessing=False  ):
        """
        Initialize the image giver.
        
        Args:
            image_urls: List of image URLs
            timeout: Timeout for image download in seconds
            num_threads: Number of worker threads/processes for downloading images
            queue_size: Maximum size of the image queue
            cache_size: Number of images to keep in cache as fallback
            use_multiprocessing: If True, use multiprocessing instead of threading
        """
        self.image_urls = image_urls
        self.timeout = timeout
        self.num_threads = num_threads
        self.cache_size = cache_size
        self.use_multiprocessing = use_multiprocessing
        
        if len(self.image_urls) == 0:
            raise ValueError("image_urls list is empty")
        
        # Queue and threading/multiprocessing setup
        if use_multiprocessing:
            self.manager = Manager()
            self.image_queue = self.manager.Queue(maxsize=queue_size)
            self.stop_event = self.manager.Event()
            self.cache_lock = self.manager.Lock()
            self.cache = self.manager.list()
            self.workers = []
            worker_type = "processes"
        else:
            self.image_queue = Queue(maxsize=queue_size)
            self.stop_event = Event()
            self.cache_lock = Lock()
            self.cache = []
            self.threads = []
            self.workers = self.threads
            worker_type = "threads"
        
        print(f"Initialized OnlineCVImagesGiver with {len(self.image_urls)} image URLs")
        print(f"Using {'multiprocessing' if use_multiprocessing else 'multithreading'}")
        print(f"Building cache of {cache_size} images...")
        
        # Initialize cache
        self._initialize_cache()
        
        print(f"Starting {num_threads} worker {worker_type} for image downloading...")
        
        # Start worker threads/processes
        self._start_workers()
    
    def _initialize_cache(self):
        """Initialize the cache with a single image."""
        print("Pre-downloading 1 image for cache...")
        attempts = 0
        max_attempts = 10
        
        while len(self.cache) == 0 and attempts < max_attempts:
            url = random.choice(self.image_urls)
            img, processed_url = download_image(url, self.timeout)
            
            if img is not None:
                self.cache.append((img.copy(), processed_url))
                print(f"Cache initialized with 1 image from {processed_url}")
                return
            
            attempts += 1
            time.sleep(0.5)
        
        if len(self.cache) == 0:
            print("Warning: Failed to initialize cache with any images")
    
    def _start_workers(self):
        """Start worker threads or processes for downloading images."""
        self.stop_event.clear()
        for i in range(self.num_threads):
            if self.use_multiprocessing:
                worker = Process(
                    target=_mp_worker_function,
                    args=(i, self.image_urls, self.timeout, self.image_queue, 
                          self.stop_event, self.cache, self.cache_lock, self.cache_size),
                    daemon=True
                )
                self.workers.append(worker)
            else:
                thread = Thread(target=self._image_generator_worker, args=(i,), daemon=True)
                thread.start()
                self.threads.append(thread)
        
        # Start multiprocessing workers if needed
        if self.use_multiprocessing:
            for worker in self.workers:
                worker.start()
    
    def _image_generator_worker(self, worker_id):
        """
        Worker function that continuously downloads images and puts them in the queue.
        
        Args:
            worker_id: ID of the worker thread
        """
        while not self.stop_event.is_set():
            try:
                # Randomly select a URL
                url = random.choice(self.image_urls)
                
                # Download image
                img, processed_url = download_image(url, self.timeout)
                
                if img is not None:
                    # print(f"Worker {worker_id} downloaded image from {processed_url}")
                    
                    # Update cache logic
                    with self.cache_lock:
                        if len(self.cache) < self.cache_size:
                            # Cache not full yet, just append
                            self.cache.append((img.copy(), processed_url))
                            # print(f"Worker {worker_id} added to cache [{len(self.cache)}/{self.cache_size}]: {processed_url}")
                        elif random.random() < 0.1:
                            # Cache is full, randomly update (10% chance)
                            cache_idx = random.randint(0, len(self.cache) - 1)
                            old_url = self.cache[cache_idx][1]
                            self.cache[cache_idx] = (img.copy(), processed_url)
                            # print(f"Worker {worker_id} updated cache[{cache_idx}]: {old_url} -> {processed_url}")
                    
                    # Put in queue (blocks if queue is full until space is available)
                    self.image_queue.put((img, processed_url))
                    # Add delay after successful download to avoid rate limiting
                    time.sleep(1)
                else:
                    # If download failed, wait before trying again
                    time.sleep(2)
                    
            except Exception as e:
                if not self.stop_event.is_set():
                    print(f"Worker {worker_id} error: {str(e)}")
                time.sleep(2)  # Longer pause on error
    
    def get(self):
        """
        Get an OpenCV image and its URL from the queue.
        Falls back to cache immediately if queue is empty.
        
        Returns:
            tuple: (opencv_image, url)
            
        Raises:
            RuntimeError: If failed to get image from queue and cache is empty
        """
        if not self.image_queue.empty():
            img, url = self.image_queue.get_nowait()
            return img, url
        else:
            # Queue is empty, get from cache
            with self.cache_lock:
                if len(self.cache) > 0:
                    cached_img, cached_url = random.choice(self.cache)
                    print(f"Queue empty, returning cached image from {cached_url}")
                    return cached_img.copy(), cached_url
                else:
                    raise RuntimeError("Failed to get image: queue is empty and cache is empty")
    
    def stop(self):
        """Stop all worker threads/processes and clean up."""
        print(f"Stopping image generator {'processes' if self.use_multiprocessing else 'threads'}...")
        self.stop_event.set()
        
        # Wait for workers to finish
        workers_to_join = self.workers if self.use_multiprocessing else self.threads
        for worker in workers_to_join:
            worker.join(timeout=2)
            if self.use_multiprocessing and worker.is_alive():
                worker.terminate()
        
        # Clear the queue
        while not self.image_queue.empty():
            try:
                self.image_queue.get_nowait()
            except:
                break
        
        if self.use_multiprocessing:
            self.workers = []
            if hasattr(self, 'manager'):
                self.manager.shutdown()
        else:
            self.threads = []
        print("All workers stopped")
    
    def __del__(self):
        """Cleanup when object is deleted."""
        self.stop()

# Multiprocessing worker function (must be at module level to be pickled)
def _mp_worker_function(worker_id, image_urls, timeout, image_queue, stop_event, cache, cache_lock, cache_size):
    """
    Worker function for multiprocessing that continuously downloads images.
    
    Args:
        worker_id: ID of the worker process
        image_urls: List of image URLs
        timeout: Timeout for downloads
        image_queue: Shared queue for images
        stop_event: Event to signal stopping
        cache: Shared cache list
        cache_lock: Lock for cache access
        cache_size: Maximum cache size
    """
    while not stop_event.is_set():
        try:
            # Randomly select a URL
            url = random.choice(image_urls)
            
            # Download image
            img, processed_url = download_image(url, timeout)
            
            if img is not None:
                # Update cache logic
                with cache_lock:
                    if len(cache) < cache_size:
                        # Cache not full yet, just append
                        cache.append((img.copy(), processed_url))
                    elif random.random() < 0.1:
                        # Cache is full, randomly update (10% chance)
                        cache_idx = random.randint(0, len(cache) - 1)
                        cache[cache_idx] = (img.copy(), processed_url)
                
                # Put in queue (blocks if queue is full until space is available)
                image_queue.put((img, processed_url))
                # Add delay after successful download to avoid rate limiting
                time.sleep(1)
            else:
                # If download failed, wait before trying again
                time.sleep(2)
                
        except Exception as e:
            if not stop_event.is_set():
                print(f"Worker {worker_id} error: {str(e)}")
            time.sleep(2)  # Longer pause on error