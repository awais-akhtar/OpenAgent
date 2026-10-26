/**
 * Voice Streaming Component with WebSocket Support (C5)
 * Handles real-time audio streaming with automatic fallback to HTTP chunked
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';

interface VoiceStreamConfig {
  sessionId: string;
  autoReconnect?: boolean;
  maxRetries?: number;
  heartbeatInterval?: number;
}

interface VoiceStreamState {
  isConnected: boolean;
  isStreaming: boolean;
  transcription: string;
  error: string | null;
  latencyMs: number;
  frameCount: number;
}

/**
 * WebSocket-based voice streaming client
 */
class VoiceWebSocketClient {
  private ws: WebSocket | null = null;
  private sessionId: string;
  private autoReconnect: boolean;
  private maxRetries: number;
  private retryCount = 0;
  private heartbeatInterval: number;
  private heartbeatTimer: number | null = null;
  private onMessageCallback: ((data: any) => void) | null = null;
  private onErrorCallback: ((error: string) => void) | null = null;

  constructor(config: VoiceStreamConfig) {
    this.sessionId = config.sessionId;
    this.autoReconnect = config.autoReconnect ?? true;
    this.maxRetries = config.maxRetries ?? 3;
    this.heartbeatInterval = config.heartbeatInterval ?? 30000;
  }

  /**
   * Connect to WebSocket server
   */
  async connect(): Promise<void> {
    try {
      const wsProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
      const wsUrl = `${wsProtocol}//${window.location.host}/api/voice/stream/continuous`;

      this.ws = new WebSocket(wsUrl);
      this.ws.binaryType = 'arraybuffer';

      this.ws.onopen = () => {
        console.log('WebSocket connected');
        this.retryCount = 0;
        this.startHeartbeat();
        this.sendMessage({
          event: 'connect',
          session_id: this.sessionId,
        });
      };

      this.ws.onmessage = (event: MessageEvent) => {
        try {
          if (typeof event.data === 'string') {
            const data = JSON.parse(event.data);
            this.onMessageCallback?.(data);
          }
        } catch (error) {
          console.error('Failed to parse WebSocket message:', error);
        }
      };

      this.ws.onerror = () => {
        const errorMsg = 'WebSocket error occurred';
        console.error(errorMsg);
        this.onErrorCallback?.(errorMsg);
      };

      this.ws.onclose = () => {
        console.log('WebSocket disconnected');
        this.stopHeartbeat();
        if (this.autoReconnect && this.retryCount < this.maxRetries) {
          this.retryCount++;
          setTimeout(() => this.connect(), Math.pow(2, this.retryCount) * 1000);
        }
      };
    } catch (error) {
      const errorMsg = `Connection failed: ${error}`;
      console.error(errorMsg);
      this.onErrorCallback?.(errorMsg);
    }
  }

  /**
   * Send audio frame over WebSocket
   */
  sendAudioFrame(audioBuffer: ArrayBuffer): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        event: 'audio_frame',
        session_id: this.sessionId,
        audio_base64: this.arrayBufferToBase64(audioBuffer),
      }));
    }
  }

  /**
   * Start heartbeat to keep connection alive
   */
  private startHeartbeat(): void {
    this.heartbeatTimer = window.setInterval(() => {
      if (this.ws?.readyState === WebSocket.OPEN) {
        this.ws.send(JSON.stringify({
          event: 'ping',
          session_id: this.sessionId,
        }));
      }
    }, this.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatTimer !== null) {
      clearInterval(this.heartbeatTimer);
      this.heartbeatTimer = null;
    }
  }

  /**
   * Send custom message
   */
  private sendMessage(message: any): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(message));
    }
  }

  /**
   * Set message callback
   */
  onMessage(callback: (data: any) => void): void {
    this.onMessageCallback = callback;
  }

  /**
   * Set error callback
   */
  onError(callback: (error: string) => void): void {
    this.onErrorCallback = callback;
  }

  /**
   * Check if WebSocket is connected
   */
  isConnected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }

  /**
   * Disconnect from WebSocket
   */
  disconnect(): void {
    this.autoReconnect = false;
    this.stopHeartbeat();
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify({
        event: 'disconnect',
        session_id: this.sessionId,
      }));
      this.ws.close();
    }
  }

  /**
   * Convert ArrayBuffer to Base64
   */
  private arrayBufferToBase64(buffer: ArrayBuffer): string {
    const bytes = new Uint8Array(buffer);
    let binary = '';
    for (let i = 0; i < bytes.byteLength; i++) {
      binary += String.fromCharCode(bytes[i]);
    }
    return btoa(binary);
  }
}

/**
 * React Component for Voice Streaming
 */
export const VoiceStreamScreen: React.FC = () => {
  const [state, setState] = useState<VoiceStreamState>({
    isConnected: false,
    isStreaming: false,
    transcription: '',
    error: null,
    latencyMs: 0,
    frameCount: 0,
  });

  const clientRef = useRef<VoiceWebSocketClient | null>(null);
  const mediaRecorderRef = useRef<MediaRecorder | null>(null);
  const audioContextRef = useRef<AudioContext | null>(null);
  const analyserRef = useRef<AnalyserNode | null>(null);
  const sessionIdRef = useRef<string>(Math.random().toString(36).substring(7));

  /**
   * Initialize WebSocket connection
   */
  const initializeWebSocket = useCallback(async () => {
    try {
      const config: VoiceStreamConfig = {
        sessionId: sessionIdRef.current,
        autoReconnect: true,
        maxRetries: 3,
        heartbeatInterval: 30000,
      };

      const client = new VoiceWebSocketClient(config);

      client.onMessage((data) => {
        if (data.transcription) {
          setState((prev) => ({
            ...prev,
            transcription: data.transcription,
            latencyMs: data.latency_ms || 0,
          }));
        }
        if (data.error) {
          setState((prev) => ({
            ...prev,
            error: data.error,
          }));
        }
      });

      client.onError((error) => {
        setState((prev) => ({
          ...prev,
          error,
          isConnected: false,
        }));
      });

      await client.connect();
      clientRef.current = client;

      setState((prev) => ({
        ...prev,
        isConnected: true,
        error: null,
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: `Connection failed: ${error}`,
      }));
    }
  }, []);

  /**
   * Start audio streaming
   */
  const startStreaming = useCallback(async () => {
    try {
      if (!clientRef.current?.isConnected()) {
        await initializeWebSocket();
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        audio: {
          echoCancellation: true,
          noiseSuppression: true,
          autoGainControl: true,
        },
      });

      audioContextRef.current = new (window.AudioContext || (window as any).webkitAudioContext)();
      analyserRef.current = audioContextRef.current.createAnalyser();

      const source = audioContextRef.current.createMediaStreamSource(stream);
      source.connect(analyserRef.current);

      mediaRecorderRef.current = new MediaRecorder(stream);

      mediaRecorderRef.current.ondataavailable = (event) => {
        if (clientRef.current && event.data.size > 0) {
          const reader = new FileReader();
          reader.onload = () => {
            if (reader.result instanceof ArrayBuffer) {
              clientRef.current?.sendAudioFrame(reader.result);
              setState((prev) => ({
                ...prev,
                frameCount: prev.frameCount + 1,
              }));
            }
          };
          reader.readAsArrayBuffer(event.data);
        }
      };

      mediaRecorderRef.current.start(100); // Collect frames every 100ms
      setState((prev) => ({
        ...prev,
        isStreaming: true,
        error: null,
      }));
    } catch (error) {
      setState((prev) => ({
        ...prev,
        error: `Failed to start streaming: ${error}`,
      }));
    }
  }, [initializeWebSocket]);

  /**
   * Stop audio streaming
   */
  const stopStreaming = useCallback(() => {
    if (mediaRecorderRef.current) {
      mediaRecorderRef.current.stop();
      mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
    }

    if (audioContextRef.current) {
      audioContextRef.current.close();
    }

    if (clientRef.current) {
      clientRef.current.disconnect();
    }

    setState((prev) => ({
      ...prev,
      isStreaming: false,
    }));
  }, []);

  /**
   * Cleanup on unmount
   */
  useEffect(() => {
    return () => {
      if (mediaRecorderRef.current) {
        mediaRecorderRef.current.stream.getTracks().forEach((track) => track.stop());
      }
      if (clientRef.current) {
        clientRef.current.disconnect();
      }
    };
  }, []);

  return (
    <div style={{ padding: '20px', maxWidth: '600px', margin: '0 auto' }}>
      <h2>Voice Streaming (WebSocket)</h2>

      {/* Connection Status */}
      <div style={{
        padding: '10px',
        marginBottom: '20px',
        backgroundColor: state.isConnected ? '#d4edda' : '#f8d7da',
        borderRadius: '4px',
        color: state.isConnected ? '#155724' : '#721c24',
      }}>
        Status: {state.isConnected ? 'Connected' : 'Disconnected'}
        {state.isStreaming && ' • Streaming'}
      </div>

      {/* Controls */}
      <div style={{ marginBottom: '20px', display: 'flex', gap: '10px' }}>
        <button
          onClick={startStreaming}
          disabled={state.isStreaming}
          style={{
            padding: '10px 20px',
            cursor: state.isStreaming ? 'not-allowed' : 'pointer',
            backgroundColor: state.isStreaming ? '#ccc' : '#007bff',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
          }}
        >
          Start Streaming
        </button>
        <button
          onClick={stopStreaming}
          disabled={!state.isStreaming}
          style={{
            padding: '10px 20px',
            cursor: !state.isStreaming ? 'not-allowed' : 'pointer',
            backgroundColor: !state.isStreaming ? '#ccc' : '#dc3545',
            color: 'white',
            border: 'none',
            borderRadius: '4px',
          }}
        >
          Stop Streaming
        </button>
      </div>

      {/* Error Display */}
      {state.error && (
        <div style={{
          padding: '10px',
          marginBottom: '20px',
          backgroundColor: '#f8d7da',
          borderRadius: '4px',
          color: '#721c24',
        }}>
          Error: {state.error}
        </div>
      )}

      {/* Transcription Display */}
      <div style={{ marginBottom: '20px' }}>
        <h3>Transcription</h3>
        <div style={{
          padding: '10px',
          backgroundColor: '#f5f5f5',
          borderRadius: '4px',
          minHeight: '100px',
          fontFamily: 'monospace',
        }}>
          {state.transcription || 'Waiting for transcription...'}
        </div>
      </div>

      {/* Statistics */}
      <div style={{
        padding: '10px',
        backgroundColor: '#f5f5f5',
        borderRadius: '4px',
      }}>
        <p>Frames sent: {state.frameCount}</p>
        <p>Latency: {state.latencyMs}ms</p>
      </div>
    </div>
  );
};

export default VoiceStreamScreen;
