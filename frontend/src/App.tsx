import { Link, Route, Routes, Navigate } from 'react-router-dom'
import { ProjectsPage } from './pages/Projects'
import { ProjectDetailPage } from './pages/ProjectDetail'
import { ReconstructionPage } from './pages/Reconstruction'
import { SettingsPage } from './pages/Settings'

// V1 ページは全部で 6 ページになる予定 (Projects / Source / Frames / Masks /
// Reconstruction / Settings). 現状は Projects / ProjectDetail / Reconstruction / Settings.
export const App = () => (
  <div className="app">
    <header className="app-header">
      <h1>sphere-reconstruct</h1>
      <nav>
        <Link to="/projects">プロジェクト</Link>
        <Link to="/settings">設定</Link>
      </nav>
    </header>
    <main className="app-main">
      <Routes>
        <Route path="/" element={<Navigate to="/projects" replace />} />
        <Route path="/projects" element={<ProjectsPage />} />
        <Route path="/projects/:id" element={<ProjectDetailPage />} />
        <Route path="/projects/:id/reconstruction" element={<ReconstructionPage />} />
        <Route path="/settings" element={<SettingsPage />} />
      </Routes>
    </main>
  </div>
)
