import { useQuery } from '@tanstack/react-query'
import { useParams } from 'react-router-dom'
import { api } from '../api/client'
import { PointCloudViewer } from '../viewers/PointCloudViewer'

export const ReconstructionPage = () => {
  const { id = '' } = useParams<{ id: string }>()
  const { data, error, isLoading } = useQuery({
    queryKey: ['reconstruction', id],
    queryFn: () => api.getReconstruction(id),
    enabled: !!id,
    retry: false,
  })

  return (
    <div className="card">
      <h2>再構成ビューア</h2>
      {isLoading && <div>読み込み中...</div>}
      {error && (
        <div className="mono">
          再構成データがまだありません. export_dataset ステージまで実行してください.
        </div>
      )}
      {data && (
        <>
          <div className="mono" style={{ marginBottom: 8 }}>
            images {data.stats.num_images} · points {data.stats.num_points3D.toLocaleString()} ·
            track {data.stats.mean_track_length.toFixed(1)}
            {data.stats.registered_ratio != null &&
              ` · reg ${(data.stats.registered_ratio * 100).toFixed(0)}%`}
          </div>
          <PointCloudViewer projectId={id} recon={data} />
        </>
      )}
    </div>
  )
}
