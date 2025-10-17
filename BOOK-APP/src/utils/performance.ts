/**
 * Performance Optimization Utilities
 * Provides debouncing, throttling, caching, and performance monitoring
 * @module utils/performance
 */

// ==================== DEBOUNCING ====================

/**
 * Creates a debounced function that delays execution until after wait milliseconds
 * @param func - Function to debounce
 * @param wait - Milliseconds to wait
 * @returns Debounced function
 * @example
 * const debouncedSearch = debounce((query) => searchBooks(query), 300);
 */
export function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout | null = null;

  return function executedFunction(...args: Parameters<T>) {
    const later = () => {
      timeout = null;
      func(...args);
    };

    if (timeout) clearTimeout(timeout);
    timeout = setTimeout(later, wait);
  };
}

// ==================== THROTTLING ====================

/**
 * Creates a throttled function that only executes at most once per wait milliseconds
 * @param func - Function to throttle
 * @param wait - Minimum time between executions in milliseconds
 * @returns Throttled function
 * @example
 * const throttledScroll = throttle(() => handleScroll(), 100);
 */
export function throttle<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let lastTime = 0;

  return function executedFunction(...args: Parameters<T>) {
    const now = Date.now();

    if (now - lastTime >= wait) {
      lastTime = now;
      func(...args);
    }
  };
}

// ==================== MEMOIZATION ====================

/**
 * Simple memoization cache
 */
class MemoCache<K, V> {
  private cache = new Map<K, V>();
  private maxSize: number;

  constructor(maxSize: number = 100) {
    this.maxSize = maxSize;
  }

  get(key: K): V | undefined {
    return this.cache.get(key);
  }

  set(key: K, value: V): void {
    if (this.cache.size >= this.maxSize) {
      // Remove oldest entry (first item)
      const firstKey = this.cache.keys().next().value;
      this.cache.delete(firstKey);
    }
    this.cache.set(key, value);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  clear(): void {
    this.cache.clear();
  }

  get size(): number {
    return this.cache.size;
  }
}

/**
 * Creates a memoized function that caches results
 * @param func - Function to memoize
 * @param keyGenerator - Optional function to generate cache key
 * @param maxSize - Maximum cache size (default: 100)
 * @returns Memoized function
 * @example
 * const memoizedExpensiveCalc = memoize((a, b) => expensiveCalc(a, b));
 */
export function memoize<T extends (...args: any[]) => any>(
  func: T,
  keyGenerator?: (...args: Parameters<T>) => string,
  maxSize: number = 100
): T & { cache: MemoCache<string, ReturnType<T>> } {
  const cache = new MemoCache<string, ReturnType<T>>(maxSize);

  const memoized = function (...args: Parameters<T>): ReturnType<T> {
    const key = keyGenerator
      ? keyGenerator(...args)
      : JSON.stringify(args);

    if (cache.has(key)) {
      return cache.get(key)!;
    }

    const result = func(...args);
    cache.set(key, result);
    return result;
  } as T & { cache: MemoCache<string, ReturnType<T>> };

  memoized.cache = cache;
  return memoized;
}

// ==================== BATCHING ====================

/**
 * Batches multiple function calls into a single execution
 * @param func - Function that accepts an array of arguments
 * @param wait - Time to wait before executing batch
 * @returns Batched function
 * @example
 * const batchedUpdate = batch((ids) => updateBooks(ids), 100);
 */
export function batch<T>(
  func: (items: T[]) => void | Promise<void>,
  wait: number
): (item: T) => void {
  let items: T[] = [];
  let timeout: NodeJS.Timeout | null = null;

  return function (item: T) {
    items.push(item);

    if (timeout) clearTimeout(timeout);

    timeout = setTimeout(() => {
      func(items);
      items = [];
      timeout = null;
    }, wait);
  };
}

// ==================== LAZY LOADING ====================

/**
 * Creates a lazy-loaded value that's computed only when accessed
 * @param factory - Function that creates the value
 * @returns Lazy value getter
 * @example
 * const lazyPDF = lazy(() => loadPDF(url));
 * const pdf = lazyPDF(); // Computed only on first call
 */
export function lazy<T>(factory: () => T): () => T {
  let value: T | undefined;
  let computed = false;

  return () => {
    if (!computed) {
      value = factory();
      computed = true;
    }
    return value!;
  };
}

// ==================== ASYNC OPERATIONS ====================

/**
 * Retries an async operation with exponential backoff
 * @param operation - Async function to retry
 * @param maxAttempts - Maximum number of attempts (default: 3)
 * @param initialDelay - Initial delay in ms (default: 1000)
 * @returns Result of the operation
 * @example
 * const data = await retry(() => fetchData(), 5, 500);
 */
export async function retry<T>(
  operation: () => Promise<T>,
  maxAttempts: number = 3,
  initialDelay: number = 1000
): Promise<T> {
  let lastError: Error;

  for (let attempt = 0; attempt < maxAttempts; attempt++) {
    try {
      return await operation();
    } catch (error) {
      lastError = error as Error;

      if (attempt < maxAttempts - 1) {
        const delay = initialDelay * Math.pow(2, attempt);
        await sleep(delay);
      }
    }
  }

  throw lastError!;
}

/**
 * Sleep for specified milliseconds
 * @param ms - Milliseconds to sleep
 * @returns Promise that resolves after delay
 */
export function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

/**
 * Timeout wrapper for promises
 * @param promise - Promise to wrap
 * @param timeoutMs - Timeout in milliseconds
 * @param errorMessage - Error message for timeout
 * @returns Promise that rejects on timeout
 * @example
 * const data = await timeout(fetchData(), 5000, 'Request timed out');
 */
export async function timeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errorMessage: string = 'Operation timed out'
): Promise<T> {
  return Promise.race([
    promise,
    new Promise<T>((_, reject) =>
      setTimeout(() => reject(new Error(errorMessage)), timeoutMs)
    ),
  ]);
}

// ==================== CHUNK PROCESSING ====================

/**
 * Process array in chunks to avoid blocking
 * @param items - Array of items to process
 * @param chunkSize - Size of each chunk
 * @param processor - Function to process each chunk
 * @param delayMs - Delay between chunks in ms
 * @example
 * await processInChunks(books, 50, (chunk) => indexBooks(chunk), 10);
 */
export async function processInChunks<T>(
  items: T[],
  chunkSize: number,
  processor: (chunk: T[]) => void | Promise<void>,
  delayMs: number = 0
): Promise<void> {
  for (let i = 0; i < items.length; i += chunkSize) {
    const chunk = items.slice(i, i + chunkSize);
    await processor(chunk);

    if (delayMs > 0 && i + chunkSize < items.length) {
      await sleep(delayMs);
    }
  }
}

// ==================== PERFORMANCE MONITORING ====================

/**
 * Measures execution time of a function
 * @param name - Name of the operation
 * @param func - Function to measure
 * @returns Result of the function
 * @example
 * const result = await measurePerformance('PDF Load', () => loadPDF());
 */
export async function measurePerformance<T>(
  name: string,
  func: () => T | Promise<T>
): Promise<T> {
  const start = performance.now();

  try {
    const result = await func();
    const duration = performance.now() - start;
    console.log(`[Performance] ${name}: ${duration.toFixed(2)}ms`);
    return result;
  } catch (error) {
    const duration = performance.now() - start;
    console.error(
      `[Performance] ${name} failed after ${duration.toFixed(2)}ms`,
      error
    );
    throw error;
  }
}

/**
 * Creates a performance tracker for multiple operations
 * @example
 * const tracker = createPerformanceTracker('PDF Processing');
 * tracker.start('Load');
 * // ... operation ...
 * tracker.end('Load');
 * tracker.report();
 */
export function createPerformanceTracker(name: string) {
  const metrics = new Map<string, number[]>();
  const startTimes = new Map<string, number>();

  return {
    start(operation: string) {
      startTimes.set(operation, performance.now());
    },

    end(operation: string) {
      const startTime = startTimes.get(operation);
      if (!startTime) {
        console.warn(`No start time found for operation: ${operation}`);
        return;
      }

      const duration = performance.now() - startTime;
      const existing = metrics.get(operation) || [];
      existing.push(duration);
      metrics.set(operation, existing);
      startTimes.delete(operation);
    },

    report() {
      console.log(`\n[Performance Report] ${name}`);
      console.log('─'.repeat(50));

      metrics.forEach((durations, operation) => {
        const avg =
          durations.reduce((a, b) => a + b, 0) / durations.length;
        const min = Math.min(...durations);
        const max = Math.max(...durations);

        console.log(`${operation}:`);
        console.log(
          `  Avg: ${avg.toFixed(2)}ms | Min: ${min.toFixed(2)}ms | Max: ${max.toFixed(2)}ms | Count: ${durations.length}`
        );
      });

      console.log('─'.repeat(50));
    },

    clear() {
      metrics.clear();
      startTimes.clear();
    },

    getMetrics() {
      return Object.fromEntries(metrics);
    },
  };
}

// ==================== MEMORY MANAGEMENT ====================

/**
 * Creates a weak cache that automatically removes entries when memory is needed
 */
export class WeakValueCache<K extends object, V> {
  private cache = new WeakMap<K, V>();
  private keys: K[] = [];
  private maxSize: number;

  constructor(maxSize: number = 50) {
    this.maxSize = maxSize;
  }

  set(key: K, value: V): void {
    if (this.keys.length >= this.maxSize) {
      const oldKey = this.keys.shift();
      if (oldKey) {
        this.cache.delete(oldKey);
      }
    }

    this.cache.set(key, value);
    this.keys.push(key);
  }

  get(key: K): V | undefined {
    return this.cache.get(key);
  }

  has(key: K): boolean {
    return this.cache.has(key);
  }

  delete(key: K): void {
    this.cache.delete(key);
    this.keys = this.keys.filter((k) => k !== key);
  }

  clear(): void {
    this.keys.forEach((key) => this.cache.delete(key));
    this.keys = [];
  }

  get size(): number {
    return this.keys.length;
  }
}

// ==================== EXPORT ====================

export { MemoCache };
