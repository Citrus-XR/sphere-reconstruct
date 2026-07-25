import { Component, type ReactNode } from 'react'

// パネル単位のエラーバウンダリ. 1 つのパネルで例外が出ても UI 全体を巻き込まない.
interface Props { children: ReactNode; label?: string }
interface State { err: Error | null }

export class ErrorBoundary extends Component<Props, State> {
  state: State = { err: null }
  static getDerivedStateFromError(err: Error): State { return { err } }
  componentDidCatch(err: Error) { console.error('panel error:', err) }
  render() {
    if (this.state.err) {
      return (
        <div className="error" style={{ padding: 12, fontSize: 12 }}>
          <div style={{ marginBottom: 8 }}>⚠ {this.props.label ?? 'panel'}: {String(this.state.err.message || this.state.err)}</div>
          <button className="btn btn-secondary" onClick={() => this.setState({ err: null })}>再試行</button>
        </div>
      )
    }
    return this.props.children
  }
}
