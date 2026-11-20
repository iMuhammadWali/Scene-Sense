export type StageInfo = {
  stage: string
  status?: string
  input_count?: number
  output_count?: number
  rejected_count?: number
  threshold?: number
  processing_time_seconds?: number
  candidate_frame_count?: number
  retained_frames?: number
  stored_images?: number
  embeddings?: number
  faiss_vectors?: number
  embedding_dim?: number
  extracted_fps?: number
  original_fps?: number
  video_duration_seconds?: number
  pruning_percentage?: number
  [key: string]: unknown
}

export type StatusPayload = {
  session_id: string
  status: string
  stage?: string
  progress: number
  message?: string
  video_ready: boolean
  stages: StageInfo[]
  error?: string | null
  video_metadata?: Record<string, unknown>
  retained_frame_count?: number
  candidate_frame_count?: number
}

export type EvidenceFrame = {
  frame_id?: number | null
  timestamp: number
  similarity?: number | null
  source?: string
}

export type QuestionResult = {
  question_id: string
  query: string
  answer: string
  query_type: 'SIMPLE' | 'COMPLEX'
  path: 'FAST' | 'SLOW'
  vlm: string
  evidence: EvidenceFrame[]
  retrieval: Record<string, unknown>
  timings: {
    stages: Record<string, number>
    total_seconds: number
    vlm_called?: boolean
    num_evidence_frames?: number
  }
  debug: Record<string, unknown>
  vlm_info?: {
    used: boolean
    provider?: string | null
    model?: string | null
    [key: string]: unknown
  }
}

export type QueryProgress = {
  session_id?: string
  question?: string
  status: string
  stage?: string | null
  progress: number
  message?: string
  stages: StageInfo[]
  error?: string | null
  query_type?: string | null
  path?: string | null
  started_at?: string
  updated_at?: string
  result_question_id?: string | null
}
