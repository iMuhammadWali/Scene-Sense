import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  askQuestion,
  evidenceUrl,
  getDebug,
  getHealth,
  getModelsStatus,
  getQueryProgress,
  getStatus,
  listQuestions,
  streamUrl,
  uploadVideo,
} from './api'
import type { QueryProgress, QuestionResult, StageInfo, StatusPayload } from './types'
import './App.css'

const PIPELINE_STAGES = [
  { key: 'upload', label: 'Video Upload' },
  { key: 'frame_extraction', label: 'Frame Extraction' },
  { key: 'clip_encoding', label: 'CLIP Encoding' },
  { key: 'frame_pruning', label: 'Frame Pruning' },
  { key: 'visual_memory', label: 'Visual Memory' },
  { key: 'faiss_index', label: 'FAISS Index' },
  { key: 'video_ready', label: 'VIDEO READY' },
]

const QUERY_STAGES = [
  { key: 'query_classification', label: 'Classify SIMPLE / COMPLEX' },
  { key: 'clip_text_embedding', label: 'CLIP Text Embedding' },
  { key: 'faiss_search', label: 'FAISS Top-20' },
  { key: 'similarity_filtering', label: 'Similarity Filtering' },
  { key: 'temporal_grouping', label: 'Temporal Grouping' },
  { key: 'diversity_selection', label: 'Diversity Selection' },
  { key: 'evidence_frame_extraction', label: 'Evidence Extraction' },
  { key: 'vlm_inference', label: 'OpenRouter VLM (COMPLEX only)' },
  { key: 'answer_ready', label: 'DONE' },
]

function stageMap(stages: StageInfo[] = []) {
  const map = new Map<string, StageInfo>()
  for (const s of stages) map.set(s.stage, s)
  return map
}

function formatBytes(n?: number) {
  if (!n && n !== 0) return '—'
  if (n < 1024) return `${n} B`
  if (n < 1024 ** 2) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 ** 2).toFixed(2)} MB`
}

function formatSeconds(n?: number | null) {
  if (n == null || Number.isNaN(n)) return null
  if (n < 60) return `${n.toFixed(2)}s`
  const m = Math.floor(n / 60)
  const s = n - m * 60
  return `${m}m ${s.toFixed(1)}s`
}

export default function App() {
  const videoRef = useRef<HTMLVideoElement>(null)
  const [researchMode, setResearchMode] = useState(true)
  const [health, setHealth] = useState<Record<string, unknown> | null>(null)
  const [modelStatus, setModelStatus] = useState<Record<string, unknown> | null>(null)
  const [sessionId, setSessionId] = useState<string | null>(null)
  const [status, setStatus] = useState<StatusPayload | null>(null)
  const [debug, setDebug] = useState<Record<string, unknown> | null>(null)
  const [fileMeta, setFileMeta] = useState<{ name: string; size: number } | null>(null)
  const [question, setQuestion] = useState('')
  const [asking, setAsking] = useState(false)
  const [queryProgress, setQueryProgress] = useState<QueryProgress | null>(null)
  const [askElapsed, setAskElapsed] = useState(0)
  const [active, setActive] = useState<QuestionResult | null>(null)
  const [history, setHistory] = useState<QuestionResult[]>([])
  const [error, setError] = useState<string | null>(null)
  const [devLog, setDevLog] = useState<string[]>([])
  const [uploading, setUploading] = useState(false)

  const log = useCallback((msg: string) => {
    setDevLog((prev) => [`${new Date().toLocaleTimeString()} — ${msg}`, ...prev].slice(0, 80))
  }, [])

  useEffect(() => {
    getHealth()
      .then((h) => {
        setHealth(h)
        log('Health endpoint OK')
      })
      .catch((e) => {
        setError(String(e.message || e))
        log(`Health failed: ${e}`)
      })
    getModelsStatus()
      .then((m) => setModelStatus(m))
      .catch((e) => log(`Models status failed: ${e}`))
  }, [log])

  useEffect(() => {
    if (!sessionId) return
    let alive = true
    const tick = async () => {
      try {
        const s = await getStatus(sessionId)
        if (!alive) return
        setStatus(s)
        if (s.video_ready || s.status === 'error') {
          const d = await getDebug(sessionId)
          if (alive) setDebug(d)
          const q = await listQuestions(sessionId)
          if (alive) setHistory(q.questions || [])
        }
      } catch (e: unknown) {
        const msg = e instanceof Error ? e.message : String(e)
        if (alive) {
          setError(msg)
          log(msg)
        }
      }
    }
    tick()
    const id = window.setInterval(tick, 1500)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [sessionId, log])

  // Live poll query pipeline while a question is running
  useEffect(() => {
    if (!sessionId || !asking) return
    let alive = true
    const tick = async () => {
      try {
        const p = (await getQueryProgress(sessionId)) as QueryProgress
        if (alive) setQueryProgress(p)
      } catch {
        /* ignore transient poll errors */
      }
    }
    tick()
    const id = window.setInterval(tick, 500)
    return () => {
      alive = false
      window.clearInterval(id)
    }
  }, [sessionId, asking])

  // Elapsed wall clock while asking
  useEffect(() => {
    if (!asking) {
      setAskElapsed(0)
      return
    }
    const t0 = performance.now()
    const id = window.setInterval(() => {
      setAskElapsed((performance.now() - t0) / 1000)
    }, 250)
    return () => window.clearInterval(id)
  }, [asking])

  const stages = useMemo(() => stageMap(status?.stages || []), [status])
  const queryStages = useMemo(
    () => stageMap(queryProgress?.stages || []),
    [queryProgress],
  )
  const visualSummary = (debug?.visual_memory_summary || {}) as Record<string, number>
  const videoReady = Boolean(status?.video_ready)

  async function onUpload(file: File | null) {
    if (!file) return
    setError(null)
    setUploading(true)
    setFileMeta({ name: file.name, size: file.size })
    setActive(null)
    setHistory([])
    setDebug(null)
    try {
      log(`Uploading ${file.name}`)
      const res = await uploadVideo(file)
      setSessionId(res.session_id)
      log(`Session ${res.session_id} created; processing started`)
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      log(`Upload error: ${msg}`)
    } finally {
      setUploading(false)
    }
  }

  async function onAsk() {
    if (!sessionId || !question.trim()) return
    setAsking(true)
    setError(null)
    setQueryProgress({
      status: 'running',
      stage: 'starting',
      progress: 0,
      message: 'Starting query pipeline…',
      stages: [],
      question: question.trim(),
    })
    try {
      log(`Asking: ${question}`)
      const res = (await askQuestion(sessionId, question.trim())) as QuestionResult
      setActive(res)
      setHistory((h) => [...h, res])
      try {
        const p = (await getQueryProgress(sessionId)) as QueryProgress
        setQueryProgress(p)
      } catch {
        /* keep last progress */
      }
      log(
        `Answered via ${res.path} / ${res.query_type} in ${res.timings?.total_seconds ?? '?'}s`,
      )
    } catch (e: unknown) {
      const msg = e instanceof Error ? e.message : String(e)
      setError(msg)
      log(`Question error: ${msg}`)
      try {
        if (sessionId) {
          const p = (await getQueryProgress(sessionId)) as QueryProgress
          setQueryProgress(p)
        }
      } catch {
        /* ignore */
      }
    } finally {
      setAsking(false)
    }
  }

  function seekTo(ts: number) {
    const v = videoRef.current
    if (!v) return
    v.currentTime = Math.max(0, ts)
    void v.play()
  }

  const topHits = ((active?.retrieval?.top_hits || []) as Array<Record<string, unknown>>)
  const groups = ((active?.retrieval?.temporal_groups || []) as Array<Record<string, unknown>>)
  const hw = (health?.hardware || {}) as Record<string, unknown>
  const ms = modelStatus || ((health?.models as Record<string, unknown>) || {})
  const clipStatus = (ms.clip || {}) as Record<string, unknown>
  const clfStatus = (ms.classifier || {}) as Record<string, unknown>
  const vlmStatus = (ms.vlm || ms.fast_vlm || {}) as Record<string, unknown>
  const vlmDebug = (active?.debug?.vlm || {}) as Record<string, unknown>
  const candidateCount = status?.candidate_frame_count
  const retainedCount = status?.retained_frame_count
  const topHitCount = topHits.length
  const finalEvidenceCount = active?.evidence?.length ?? 0

  return (
    <div className="app">
      <header className="topbar">
        <div>
          <h1>SceneSense</h1>
          <p>Efficient Video Question Answering — Research MVP</p>
        </div>
        <label className="toggle">
          <input
            type="checkbox"
            checked={researchMode}
            onChange={(e) => setResearchMode(e.target.checked)}
          />
          Research / Debug Mode
        </label>
      </header>

      {error && <div className="banner error">{error}</div>}

      <section className="grid-top">
        <div className="panel">
          <h2>Video</h2>
          {!sessionId && (
            <div
              className="dropzone"
              onDragOver={(e) => e.preventDefault()}
              onDrop={(e) => {
                e.preventDefault()
                const f = e.dataTransfer.files?.[0]
                void onUpload(f || null)
              }}
            >
              <p>Drop an MP4 here or choose a file</p>
              <input
                type="file"
                accept="video/mp4,.mp4"
                disabled={uploading}
                onChange={(e) => void onUpload(e.target.files?.[0] || null)}
              />
            </div>
          )}
          {fileMeta && (
            <div className="meta-row">
              <span>{fileMeta.name}</span>
              <span>{formatBytes(fileMeta.size)}</span>
              {status?.video_metadata && (
                <span>
                  {Number((status.video_metadata as { duration_seconds?: number }).duration_seconds || 0).toFixed(1)}s
                </span>
              )}
            </div>
          )}
          {sessionId && (
            <video
              ref={videoRef}
              className="player"
              controls
              src={streamUrl(sessionId)}
            />
          )}
          {sessionId && (
            <button
              className="ghost"
              onClick={() => {
                setSessionId(null)
                setStatus(null)
                setActive(null)
                setHistory([])
                setFileMeta(null)
              }}
            >
              Upload another video
            </button>
          )}
        </div>

        <div className="panel">
          <h2>Processing Pipeline</h2>
          <div className="progress">
            <div style={{ width: `${Math.round((status?.progress || 0) * 100)}%` }} />
          </div>
          <ol className="stages">
            {PIPELINE_STAGES.map((s) => {
              const info = stages.get(s.key)
              const done =
                info?.status === 'completed' ||
                (s.key === 'video_ready' && videoReady) ||
                (s.key === 'upload' && Boolean(sessionId))
              const current = status?.stage === s.key && !videoReady
              return (
                <li key={s.key} className={done ? 'done' : current ? 'current' : ''}>
                  <div className="stage-head">
                    <strong>
                      {done ? '✓' : current ? '…' : '○'} {s.label}
                    </strong>
                    {info?.processing_time_seconds != null && (
                      <span>{info.processing_time_seconds}s</span>
                    )}
                  </div>
                  {researchMode && info && (
                    <div className="stage-body">
                      {info.extracted_fps != null && <span>FPS: {info.extracted_fps}</span>}
                      {info.input_count != null && <span>In: {info.input_count}</span>}
                      {info.output_count != null && <span>Out: {info.output_count}</span>}
                      {info.rejected_count != null && <span>Rejected: {info.rejected_count}</span>}
                      {info.threshold != null && <span>Threshold: {info.threshold}</span>}
                      {info.candidate_frame_count != null && (
                        <span>Candidates: {info.candidate_frame_count}</span>
                      )}
                      {info.stored_images != null && <span>Stored images: {info.stored_images}</span>}
                      {info.embedding_dim != null && <span>Dim: {info.embedding_dim}</span>}
                    </div>
                  )}
                </li>
              )
            })}
          </ol>
          {videoReady && <div className="ready">VIDEO READY</div>}
          {status?.status === 'error' && (
            <div className="banner error">Processing failed: {status.error}</div>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Visual Memory</h2>
        <div className="stats">
          <div>
            <em>Retained frames</em>
            <strong>{visualSummary.retained_frames ?? status?.retained_frame_count ?? '—'}</strong>
          </div>
          <div>
            <em>Stored images</em>
            <strong>{visualSummary.stored_images ?? 0}</strong>
          </div>
          <div>
            <em>Embeddings</em>
            <strong>{visualSummary.embeddings ?? '—'}</strong>
          </div>
          <div>
            <em>FAISS vectors</em>
            <strong>{visualSummary.faiss_vectors ?? '—'}</strong>
          </div>
        </div>
        <p className="hint">
          Video retrieval and indexing are performed locally. Images are not permanently stored.
          The original MP4 is the source of truth; evidence frames are extracted on demand.
        </p>
        <p className="hint">
          SIMPLE = visual evidence only (CLIP/FAISS, no VLM). COMPLEX = natural-language answer via OpenRouter VLM.
        </p>
      </section>

      <section className="panel ask">
        <h2>Ask a question</h2>
        <div className="ask-row">
          <input
            value={question}
            onChange={(e) => setQuestion(e.target.value)}
            placeholder={
              videoReady
                ? 'e.g. What did the man do after picking up the bag?'
                : 'Waiting for VIDEO READY…'
            }
            disabled={!videoReady || asking}
            onKeyDown={(e) => {
              if (e.key === 'Enter') void onAsk()
            }}
          />
          <button disabled={!videoReady || asking || !question.trim()} onClick={() => void onAsk()}>
            {asking ? 'Running pipeline…' : 'Ask'}
          </button>
        </div>
        {asking && (
          <p className="hint">
            Elapsed {formatSeconds(askElapsed)}. SIMPLE is fast (no VLM). COMPLEX calls OpenRouter —
            watch stages below.
          </p>
        )}
      </section>

      <section className="panel">
        <h2>Query Pipeline</h2>
        <div className="progress">
          <div
            style={{
              width: `${Math.round(
                (asking
                  ? queryProgress?.progress || 0.05
                  : queryProgress?.status === 'completed'
                    ? 1
                    : queryProgress?.progress || 0) * 100,
              )}%`,
            }}
          />
        </div>
        {(queryProgress?.query_type || queryProgress?.path || asking) && (
          <div className="chips" style={{ marginBottom: '0.65rem' }}>
            {queryProgress?.query_type && <span>Type: {queryProgress.query_type}</span>}
            {queryProgress?.path && <span>Path: {queryProgress.path}</span>}
            {asking && <span>Live: {queryProgress?.stage || 'starting'}</span>}
            {asking && <span>Elapsed: {formatSeconds(askElapsed)}</span>}
            {!asking && active?.timings?.total_seconds != null && (
              <span>Total: {formatSeconds(active.timings.total_seconds)}</span>
            )}
          </div>
        )}
        {queryProgress?.message && <p className="hint">{queryProgress.message}</p>}
        <ol className="stages">
          {QUERY_STAGES.map((s) => {
            const info = queryStages.get(s.key)
            const done =
              info?.status === 'completed' ||
              info?.status === 'skipped' ||
              (s.key === 'answer_ready' && queryProgress?.status === 'completed')
            const current =
              asking &&
              (queryProgress?.stage === s.key || info?.status === 'running') &&
              !done
            const timing =
              info?.status === 'skipped'
                ? 'skipped'
                : info?.processing_time_seconds != null
                  ? formatSeconds(Number(info.processing_time_seconds))
                  : current
                    ? '…'
                    : null
            return (
              <li key={s.key} className={done ? 'done' : current ? 'current' : ''}>
                <div className="stage-head">
                  <strong>
                    {done ? '✓' : current ? '…' : '○'} {s.label}
                  </strong>
                  {timing && <span>{timing}</span>}
                </div>
                {(researchMode || current || done) && info && (
                  <div className="stage-body">
                    {info.message != null && <span>{String(info.message)}</span>}
                    {info.query_type != null && <span>Type: {String(info.query_type)}</span>}
                    {info.path != null && <span>Path: {String(info.path)}</span>}
                    {info.input_count != null && <span>In: {String(info.input_count)}</span>}
                    {info.output_count != null && <span>Out: {String(info.output_count)}</span>}
                    {info.top_k != null && <span>Top-K: {String(info.top_k)}</span>}
                    {info.threshold != null && <span>Threshold: {String(info.threshold)}</span>}
                    {info.embedding_dim != null && <span>Dim: {String(info.embedding_dim)}</span>}
                    {info.num_images != null && <span>Frames: {String(info.num_images)}</span>}
                    {info.device != null && <span>Device: {String(info.device)}</span>}
                    {info.model != null && <span>Model: {String(info.model)}</span>}
                    {info.already_loaded != null && (
                      <span>{info.already_loaded ? 'Reused' : 'Cold load'}</span>
                    )}
                  </div>
                )}
              </li>
            )
          })}
        </ol>
        {queryProgress?.status === 'completed' && !asking && (
          <div className="ready">ANSWER READY</div>
        )}
        {queryProgress?.status === 'error' && (
          <div className="banner error">Query failed: {queryProgress.error}</div>
        )}
        {!asking && !queryProgress && (
          <p className="muted">Ask a question to see live retrieval + VLM stages here.</p>
        )}
      </section>

      <section className="grid-mid">
        <div className="panel">
          <h2>Answer</h2>
          {!active && <p className="muted">No answer yet.</p>}
          {active && (
            <>
              <div className="answer" style={{ whiteSpace: 'pre-wrap' }}>
                {active.path === 'FAST' ? (
                  <>
                    <div className="chips" style={{ marginBottom: '0.5rem' }}>
                      <span>Visual evidence retrieved — VLM not used.</span>
                    </div>
                    {active.answer}
                  </>
                ) : (
                  <>
                    <div className="chips" style={{ marginBottom: '0.5rem' }}>
                      <span>Natural-language answer generated by VLM.</span>
                    </div>
                    {active.answer}
                  </>
                )}
              </div>
              <div className="chips">
                <span>Type: {active.query_type}</span>
                <span>Path: {active.path}</span>
                <span>VLM: {active.vlm}</span>
                <span>
                  Provider:{' '}
                  {active.path === 'FAST'
                    ? 'none'
                    : String(vlmDebug.provider || 'openrouter')}
                </span>
                <span>
                  {active.path === 'FAST' ? 'Evidence frames' : 'Frames to VLM'}:{' '}
                  {finalEvidenceCount}
                </span>
                <span>Latency: {active.timings?.total_seconds}s</span>
              </div>
              {researchMode && (
                <div className="stats" style={{ marginTop: '0.75rem' }}>
                  <div>
                    <em>Candidates</em>
                    <strong>{candidateCount ?? '—'}</strong>
                  </div>
                  <div>
                    <em>Retained</em>
                    <strong>{retainedCount ?? '—'}</strong>
                  </div>
                  <div>
                    <em>FAISS Top</em>
                    <strong>{topHitCount || '—'}</strong>
                  </div>
                  <div>
                    <em>{active.path === 'FAST' ? 'Evidence' : 'Sent to VLM'}</em>
                    <strong>{finalEvidenceCount}</strong>
                  </div>
                </div>
              )}
              <h3>Evidence timeline</h3>
              <div className="timeline">
                {active.evidence?.map((e, i) => (
                  <button key={`${e.timestamp}-${i}`} onClick={() => seekTo(e.timestamp)}>
                    {e.timestamp.toFixed(1)}s
                  </button>
                ))}
              </div>
              {sessionId && (
                <div className="evidence-grid">
                  {active.evidence?.map((e, i) => {
                    const sourceId =
                      (e as Record<string, unknown>).source_frame_id ?? e.frame_id
                    const retrievedId = (e as Record<string, unknown>).retrieved_frame_id
                    const isFallback = Boolean((e as Record<string, unknown>).fallback)
                    return (
                      <button
                        key={`thumb-${e.timestamp}-${i}`}
                        className="evidence-card"
                        onClick={() => seekTo(e.timestamp)}
                      >
                        <img src={evidenceUrl(sessionId, e.timestamp)} alt={`t=${e.timestamp}`} />
                        <div>
                          <strong>#{i + 1}</strong> {e.timestamp.toFixed(2)}s
                          {sourceId != null && <div>Source frame {String(sourceId)}</div>}
                          {retrievedId != null && active.path === 'SLOW' && (
                            <div>Retrieved FAISS {String(retrievedId)}</div>
                          )}
                          {e.similarity != null && (
                            <div>
                              Sim: {Number(e.similarity).toFixed(3)}
                              {isFallback ? ' (fallback)' : ''}
                            </div>
                          )}
                        </div>
                      </button>
                    )
                  })}
                </div>
              )}
            </>
          )}
        </div>

        <div className="panel">
          <h2>Query Pipeline {researchMode ? '(Debug)' : ''}</h2>
          {!active && <p className="muted">Submit a question to inspect the pipeline.</p>}
          {active && researchMode && (
            <div className="debug-stack">
              <details open>
                <summary>Query Classification</summary>
                <div className="chips" style={{ marginBottom: '0.5rem' }}>
                  <span>
                    Result:{' '}
                    {String(
                      (active.debug?.classification as Record<string, unknown>)?.result ||
                        active.query_type,
                    )}
                  </span>
                  <span>
                    Simple conf:{' '}
                    {String(
                      (active.debug?.classification as Record<string, unknown>)
                        ?.simple_confidence ?? '—',
                    )}
                  </span>
                  <span>
                    Complex conf:{' '}
                    {String(
                      (active.debug?.classification as Record<string, unknown>)
                        ?.complex_confidence ?? '—',
                    )}
                  </span>
                </div>
                <pre>{JSON.stringify(active.debug?.classification, null, 2)}</pre>
              </details>
              <details open>
                <summary>CLIP Query Embedding</summary>
                <pre>{JSON.stringify(active.debug?.clip_query_embedding, null, 2)}</pre>
              </details>
              <details open>
                <summary>FAISS Top 20</summary>
                <table>
                  <thead>
                    <tr>
                      <th>Rank</th>
                      <th>Frame</th>
                      <th>Timestamp</th>
                      <th>Similarity</th>
                    </tr>
                  </thead>
                  <tbody>
                    {topHits.map((h) => (
                      <tr key={String(h.rank)}>
                        <td>{String(h.rank)}</td>
                        <td>{String(h.frame_id)}</td>
                        <td>
                          <button className="linkish" onClick={() => seekTo(Number(h.timestamp))}>
                            {Number(h.timestamp).toFixed(2)}s
                          </button>
                        </td>
                        <td>{Number(h.similarity).toFixed(4)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </details>
              <details open>
                <summary>Similarity Filtering</summary>
                {(() => {
                  const f = (active.debug?.filtering || {}) as Record<string, unknown>
                  return (
                    <div className="chips" style={{ marginBottom: '0.5rem' }}>
                      <span>Threshold: {String(f.threshold ?? '—')}</span>
                      <span>Candidates: {String(f.candidate_count ?? f.input_count ?? '—')}</span>
                      <span>Passed: {String(f.passed_count ?? '—')}</span>
                      <span>
                        Fallback:{' '}
                        {f.fallback_used ? 'BEST MATCH SELECTED' : 'NO'}
                      </span>
                      <span>Best sim: {String(f.best_similarity ?? '—')}</span>
                    </div>
                  )
                })()}
                <pre>{JSON.stringify(active.debug?.filtering, null, 2)}</pre>
              </details>
              <details open>
                <summary>Temporal Groups</summary>
                {groups.map((g) => (
                  <div key={String(g.group_id)} className="group">
                    Group {String(g.group_id)}: {String(g.start_timestamp)}s –{' '}
                    {String(g.end_timestamp)}s
                    <div>{JSON.stringify(g.timestamps)}</div>
                  </div>
                ))}
              </details>
              {active.path === 'SLOW' && (
                <details open>
                  <summary>Temporal Evidence</summary>
                  {(() => {
                    const items =
                      (
                        (active.debug?.evidence_extraction as Record<string, unknown>)
                          ?.evidence_items as Array<Record<string, unknown>>
                      ) ||
                      active.evidence?.map((e, i) => ({
                        evidence_id: i + 1,
                        timestamp: e.timestamp,
                        source_frame_id: (e as Record<string, unknown>).source_frame_id,
                        retrieved_frame_id: (e as Record<string, unknown>)
                          .retrieved_frame_id,
                      })) ||
                      []
                    return (
                      <ol className="stages">
                        {items.map((it) => (
                          <li key={String(it.evidence_id ?? it.timestamp)}>
                            <div className="stage-head">
                              <strong>
                                #{String(it.evidence_id)}{' '}
                                <button
                                  className="linkish"
                                  onClick={() => seekTo(Number(it.timestamp))}
                                >
                                  {Number(it.timestamp).toFixed(1)}s
                                </button>
                              </strong>
                              <span>source frame {String(it.source_frame_id ?? '—')}</span>
                            </div>
                            <div className="stage-body">
                              {it.retrieved_frame_id != null && (
                                <span>
                                  Triggered by FAISS frame {String(it.retrieved_frame_id)}
                                </span>
                              )}
                            </div>
                          </li>
                        ))}
                      </ol>
                    )
                  })()}
                  <pre>
                    {JSON.stringify(active.debug?.evidence_extraction, null, 2)}
                  </pre>
                </details>
              )}
              <details open>
                <summary>Diversity / Final Evidence</summary>
                <pre>{JSON.stringify(active.debug?.diversity_selection, null, 2)}</pre>
              </details>
              <details open>
                <summary>VLM</summary>
                <div className="chips" style={{ marginBottom: '0.5rem' }}>
                  <span>
                    Used:{' '}
                    {active.path === 'FAST' ||
                    (active.debug?.vlm as Record<string, unknown>)?.used === false
                      ? 'NO'
                      : 'YES'}
                  </span>
                  <span>
                    Reason:{' '}
                    {String(
                      (active.debug?.vlm as Record<string, unknown>)?.reason ||
                        (active.path === 'FAST' ? 'SIMPLE query' : 'COMPLEX query'),
                    )}
                  </span>
                  {active.path === 'SLOW' && (
                    <>
                      <span>
                        Provider:{' '}
                        {String((active.debug?.vlm as Record<string, unknown>)?.provider || '—')}
                      </span>
                      <span>
                        Model:{' '}
                        {String(
                          (active.debug?.vlm as Record<string, unknown>)?.vlm_model_id ||
                            (active.debug?.vlm as Record<string, unknown>)?.model ||
                            '—',
                        )}
                      </span>
                    </>
                  )}
                </div>
                <pre>{JSON.stringify(active.debug?.vlm, null, 2)}</pre>
              </details>
              <details>
                <summary>Per-stage Latency</summary>
                <pre>{JSON.stringify(active.timings, null, 2)}</pre>
              </details>
              <details>
                <summary>Routing</summary>
                <pre>{JSON.stringify(active.debug?.routing, null, 2)}</pre>
              </details>
            </div>
          )}
          {active && !researchMode && (
            <div className="chips">
              <span>{active.query_type}</span>
              <span>{active.path}</span>
              <span>{active.vlm}</span>
            </div>
          )}
        </div>
      </section>

      <section className="panel">
        <h2>Query History</h2>
        {history.length === 0 && <p className="muted">No questions yet for this video session.</p>}
        <ul className="history">
          {history.map((q) => (
            <li key={q.question_id}>
              <button
                onClick={() => setActive(q)}
                className={active?.question_id === q.question_id ? 'active' : ''}
              >
                <strong>{q.query}</strong>
                <span>
                  {q.query_type} · {q.path} · {q.vlm}
                </span>
                <em>{q.answer}</em>
              </button>
            </li>
          ))}
        </ul>
      </section>

      {researchMode && (
        <section className="panel">
          <h2>Model Status</h2>
          <h3>LOCAL MODELS</h3>
          <ul className="model-list">
            <li>
              <span className={clipStatus.cached ? 'ok' : 'bad'}>
                {clipStatus.cached ? '✓' : '⚠'}
              </span>
              CLIP — {String(clipStatus.model || '—')}
              {clipStatus.loaded ? ' (loaded)' : clipStatus.cached ? ' (cached)' : ' (missing)'}
              <div className="hint">Mode: Local</div>
            </li>
            <li>
              <span className={clfStatus.cached ? 'ok' : 'bad'}>
                {clfStatus.cached ? '✓' : '⚠'}
              </span>
              Classifier — {String(clfStatus.model || '—')}
              {clfStatus.loaded ? ' (loaded)' : clfStatus.cached ? ' (cached)' : ' (missing)'}
              <div className="hint">Mode: Local · zero-shot SIMPLE/COMPLEX</div>
            </li>
            <li>
              <span className={vlmStatus.configured ? 'ok' : 'bad'}>
                {vlmStatus.configured ? '✓' : '⚠'}
              </span>
              VLM — {String(vlmStatus.model || 'google/gemini-2.5-flash')}
              {vlmStatus.configured ? ' (configured)' : ' (not configured)'}
              <div className="hint">
                Provider: OpenRouter · Mode: API · COMPLEX path only
              </div>
              <div className="hint">
                SIMPLE / FAST never calls the VLM (CLIP retrieval only).
              </div>
            </li>
          </ul>
          <div className="stats">
            <div>
              <em>Device</em>
              <strong>{String(hw.recommended_device || '—')}</strong>
            </div>
            <div>
              <em>CUDA</em>
              <strong>{String(hw.cuda_available)}</strong>
            </div>
            <div>
              <em>RAM</em>
              <strong>{hw.ram_gb != null ? `${hw.ram_gb} GB` : '—'}</strong>
            </div>
            <div>
              <em>GPU</em>
              <strong>{String(hw.gpu_name || 'none')}</strong>
            </div>
          </div>
          <h3>Developer log</h3>
          <pre className="devlog">{devLog.join('\n')}</pre>
        </section>
      )}
    </div>
  )
}
