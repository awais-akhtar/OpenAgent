import React from 'react';
import { createRoot } from 'react-dom/client';
import App from './App';
import './app.css';

const target = document.getElementById('app');
if (!target) throw new Error('OpenAgent app mount point is missing.');
createRoot(target).render(<React.StrictMode><App /></React.StrictMode>);
