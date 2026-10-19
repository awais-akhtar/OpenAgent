import React from 'react';
import ReactDOM from 'react-dom/client';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import { MotionConfig } from 'motion/react';
import { App } from './App';
import './tailwind.css';
import './reference.css';
import './execution-trace.css';
import './app.css';
import { useWorkspaceState } from './state';

const queryClient = new QueryClient({
  defaultOptions: { queries: { retry: false, refetchOnWindowFocus: false } },
});

function Root() {
  const reduceMotion = useWorkspaceState((state) => state.reduceMotion);
  return <QueryClientProvider client={queryClient}>
    <MotionConfig reducedMotion={reduceMotion ? 'always' : 'user'}>
      <BrowserRouter><App /></BrowserRouter>
    </MotionConfig>
  </QueryClientProvider>;
}

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <Root />
  </React.StrictMode>,
);
