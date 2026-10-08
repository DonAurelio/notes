-- PerfettoSQL checkpoints for traces/tiny_with_itt.pftrace.
-- In Perfetto, open "Query (SQL)", select one statement at a time, and run it.

-- 1. Which application-level ITT regions were captured?
SELECT
  name,
  COUNT(*) AS occurrences,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms,
  ROUND(AVG(dur) / 1e6, 3) AS average_ms
FROM slice
WHERE name GLOB 'LlamaDemo:*'
  AND dur >= 0
GROUP BY name
ORDER BY name;

-- 2. Compare the duration of each outer training step.
SELECT
  name,
  ROUND(dur / 1e6, 3) AS duration_ms,
  ROUND(ts / 1e6, 3) AS start_ms
FROM slice
WHERE name GLOB 'LlamaDemo:Step.*'
  AND dur >= 0
ORDER BY ts;

-- 3. Compare the major repeated phases.
SELECT
  name,
  COUNT(*) AS occurrences,
  ROUND(MIN(dur) / 1e6, 3) AS min_ms,
  ROUND(AVG(dur) / 1e6, 3) AS average_ms,
  ROUND(MAX(dur) / 1e6, 3) AS max_ms
FROM slice
WHERE name IN (
  'LlamaDemo:Forward',
  'LlamaDemo:Backward',
  'LlamaDemo:Optimizer.Step'
)
  AND dur >= 0
GROUP BY name
ORDER BY name;

-- 4. Count per-layer envelopes. With --checkpoint, block forwards are replayed
-- during Backward, so an occurrence count can exceed the training-step count.
SELECT
  name,
  COUNT(*) AS occurrences,
  ROUND(SUM(dur) / 1e6, 3) AS total_ms
FROM slice
WHERE name GLOB 'LlamaDemo:Layer.*'
  AND dur >= 0
GROUP BY name
ORDER BY name;
