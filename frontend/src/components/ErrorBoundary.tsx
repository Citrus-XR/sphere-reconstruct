import { Component, type ReactNode } from 'react'
import { useSettings } from '../ui/settings'

// パネル単位のエラーバウンダリ. 1 つのパネルで例外が出ても UI 全体を巻き込まない.
interface Props { children: ReactNode; label?: string; retryLabel: string }
interface State { err: Error | null }

class PanelErrorBoundary extends Component<Props, State> {
  state: State = { err: null }
  static getDerivedStateFromError(err: Error): State { return { err } }
  componentDidCatch(err: Error) { console.error('panel error:', err) }
  render() {
    if (this.state.err) {
      return (
        <div className="error" style={{ padding: 12, fontSize: 12 }}>
          <div style={{ marginBottom: 8 }}>⚠ {this.props.label ?? 'panel'}: {String(this.state.err.message || this.state.err)}</div>
          <button type="button" className="btn btn-secondary"
            onClick={() => this.setState({ err: null })}>{this.props.retryLabel}</button>
        </div>
      )
    }
    return this.props.children
  }
}

export const ErrorBoundary = ({ children, label }: Omit<Props, 'retryLabel'>) => {
  const { t } = useSettings()
  return <PanelErrorBoundary label={label} retryLabel={t('retry')}>{children}</PanelErrorBoundary>
}
