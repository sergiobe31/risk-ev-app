import { Component, StrictMode } from 'react'
import type { ErrorInfo, ReactNode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// Sin esto, un error de render deja la pestaña en blanco sin explicación.
class ErrorBoundary extends Component<{ children: ReactNode }, { msg: string | null }> {
  state = { msg: null as string | null }
  static getDerivedStateFromError(e: unknown) {
    return { msg: e instanceof Error ? e.message : String(e) }
  }
  componentDidCatch(e: Error, info: ErrorInfo) {
    console.error('Error de render:', e, info)
  }
  render() {
    if (this.state.msg) {
      return (
        <div className="mx-auto max-w-xl px-6 py-16">
          <h1 className="text-lg font-semibold text-rose-400">La vista crasheó</h1>
          <pre className="mt-3 overflow-x-auto rounded-md border border-zinc-800 bg-zinc-900 p-3 text-xs text-zinc-300">
            {this.state.msg}
          </pre>
          <button
            onClick={() => location.reload()}
            className="mt-4 rounded-md bg-emerald-600 px-4 py-2 text-sm font-semibold text-zinc-50 hover:bg-emerald-500"
          >
            Recargar
          </button>
        </div>
      )
    }
    return this.props.children
  }
}

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
    </ErrorBoundary>
  </StrictMode>,
)
