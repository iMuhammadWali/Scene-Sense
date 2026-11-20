const API_BASE = import.meta.env.VITE_API_BASE || '/api'

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json()
    if (typeof data.detail === 'string') return data.detail
    if (data.detail && typeof data.detail === 'object') {
      const d = data.detail
      if (d.error === 'VLM_API_ERROR' || d.error === 'VLM_ERROR') {
        return [
          'VLM ERROR',
          `Provider: ${d.provider ?? 'local'}`,
          `Model: ${d.model ?? '—'}`,
          d.path ? `Path: ${d.path}` : null,
          d.http_status != null ? `HTTP status: ${d.http_status}` : null,
          `Reason: ${d.reason ?? JSON.stringify(d)}`,
        ]
          .filter(Boolean)
          .join('\n')
      }
      return JSON.stringify(data.detail, null, 2)
    }
    return data.message || JSON.stringify(data)
  } catch {
    return res.statusText || 'Request failed'
  }
}

export async function getHealth() {
  const res = await fetch(`${API_BASE}/health`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function getModelsStatus() {
  const res = await fetch(`${API_BASE}/models/status`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function uploadVideo(file: File) {
  const form = new FormData()
  form.append('file', file)
  const res = await fetch(`${API_BASE}/videos/upload?auto_process=true`, {
    method: 'POST',
    body: form,
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function getStatus(sessionId: string) {
  const res = await fetch(`${API_BASE}/videos/${sessionId}/status`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function getDebug(sessionId: string) {
  const res = await fetch(`${API_BASE}/videos/${sessionId}/debug`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function askQuestion(sessionId: string, question: string) {
  const res = await fetch(`${API_BASE}/videos/${sessionId}/questions`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ question }),
  })
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function getQueryProgress(sessionId: string) {
  const res = await fetch(`${API_BASE}/videos/${sessionId}/query-progress`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export async function listQuestions(sessionId: string) {
  const res = await fetch(`${API_BASE}/videos/${sessionId}/questions`)
  if (!res.ok) throw new Error(await parseError(res))
  return res.json()
}

export function streamUrl(sessionId: string) {
  return `${API_BASE}/videos/${sessionId}/stream`
}

export function evidenceUrl(sessionId: string, timestamp: number) {
  return `${API_BASE}/videos/${sessionId}/evidence/${timestamp}`
}
