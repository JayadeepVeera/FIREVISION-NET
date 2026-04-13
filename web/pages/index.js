import { useEffect, useRef, useState } from "react";

const API_URL = process.env.NEXT_PUBLIC_API_URL;

export default function Home() {
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);
  const audioContextRef = useRef(null);
  const lastSoundStatusRef = useRef(null);

  const [streaming, setStreaming] = useState(false);
  const [status, setStatus] = useState("SAFE");
  const [processedImage, setProcessedImage] = useState(null);
  const [events, setEvents] = useState([]);
  const [loading, setLoading] = useState(false);

  const fetchEvents = async () => {
    try {
      const res = await fetch(`${API_URL}/events`);
      const data = await res.json();
      setEvents(data.events || []);
    } catch (e) {
      console.error(e);
    }
  };

  const unlockAudio = async () => {
    try {
      if (!audioContextRef.current) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        audioContextRef.current = new AudioCtx();
      }

      if (audioContextRef.current.state === "suspended") {
        await audioContextRef.current.resume();
      }
    } catch (err) {
      console.warn("Audio context unlock failed:", err);
    }
  };

  const playTone = (frequency, startTime, duration, volume = 0.05) => {
    const ctx = audioContextRef.current;
    if (!ctx) return;

    const oscillator = ctx.createOscillator();
    const gainNode = ctx.createGain();

    oscillator.type = "square";
    oscillator.frequency.setValueAtTime(frequency, startTime);

    gainNode.gain.setValueAtTime(0.0001, startTime);
    gainNode.gain.exponentialRampToValueAtTime(volume, startTime + 0.01);
    gainNode.gain.exponentialRampToValueAtTime(0.0001, startTime + duration);

    oscillator.connect(gainNode);
    gainNode.connect(ctx.destination);

    oscillator.start(startTime);
    oscillator.stop(startTime + duration);
  };

  const playFireAlertSound = async (detectedStatus) => {
    const fireStatuses = ["REAL FIRE", "FAKE FIRE"];
    if (!fireStatuses.includes(detectedStatus)) return;
    if (lastSoundStatusRef.current === detectedStatus) return;

    try {
      await unlockAudio();

      const ctx = audioContextRef.current;
      if (!ctx) return;

      const now = ctx.currentTime + 0.02;

      playTone(880, now, 0.16, 0.06);
      playTone(660, now + 0.22, 0.16, 0.06);
      playTone(880, now + 0.44, 0.2, 0.07);

      lastSoundStatusRef.current = detectedStatus;
    } catch (err) {
      console.warn("Built-in fire sound failed:", err);
    }
  };

  const startCamera = async () => {
    try {
      await unlockAudio();

      const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        await videoRef.current.play();
      }

      setStreaming(true);

      intervalRef.current = setInterval(async () => {
        await captureAndSendFrame();
      }, 1500);
    } catch (err) {
      alert("Camera access failed: " + err.message);
    }
  };

  const stopCamera = () => {
    if (intervalRef.current) clearInterval(intervalRef.current);

    const stream = videoRef.current?.srcObject;
    if (stream) {
      stream.getTracks().forEach((track) => track.stop());
    }

    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }

    setStreaming(false);
    lastSoundStatusRef.current = null;
  };

  const captureAndSendFrame = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    const imageBase64 = canvas.toDataURL("image/jpeg", 0.8);

    try {
      setLoading(true);

      const res = await fetch(`${API_URL}/detect-frame`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json"
        },
        body: JSON.stringify({ image: imageBase64 })
      });

      const data = await res.json();

      if (data.ok) {
        const detectedStatus = data.status || "SAFE";

        setStatus(detectedStatus);
        setProcessedImage(data.processed_image || null);
        fetchEvents();

        if (detectedStatus === "SAFE" || detectedStatus === "REAL SMOKE" || detectedStatus === "FAKE SMOKE") {
          lastSoundStatusRef.current = null;
        }

        await playFireAlertSound(detectedStatus);
      } else {
        console.error(data.error);
      }
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  };

  const testTelegram = async () => {
    try {
      const res = await fetch(`${API_URL}/test-telegram`, {
        method: "POST"
      });
      const data = await res.json();
      alert(data.ok ? "Telegram test sent ✨" : `Telegram failed: ${data.info}`);
    } catch (err) {
      alert("Telegram test request failed: " + err.message);
    }
  };

  const clearData = async () => {
    try {
      const res = await fetch(`${API_URL}/events`, {
        method: "DELETE"
      });
      const data = await res.json();

      if (data.ok) {
        setEvents([]);
        alert("Data cleared 🧹");
      } else {
        alert("Failed to clear data");
      }
    } catch (err) {
      alert("Clear data failed: " + err.message);
    }
  };

  useEffect(() => {
    fetchEvents();
    return () => stopCamera();
  }, []);

  const statusClass =
    status === "REAL FIRE" || status === "FAKE FIRE"
      ? "status-fire"
      : status === "REAL SMOKE" || status === "FAKE SMOKE"
      ? "status-smoke"
      : "status-safe";

  return (
    <div className="page-shell">
      <header className="topbar">
        <div className="brand">
          <div className="brand-mark">🔥</div>
          <div>
            <h3>FireVisionNet</h3>
            <p>Live Monitoring</p>
          </div>
        </div>

        <div className="top-links">
          <span>Home</span>
          <span>Camera</span>
          <span>Alerts</span>
        </div>
      </header>

      <div className="hero">
        <div className="hero-badge">✨ Gentle, simple live safety dashboard</div>
        <h1>FireVisionNet</h1>
        <p>
          A sweet and clean space to watch your live camera, view detections,
          receive Telegram alerts, and keep an eye on recent events.
        </p>
      </div>

      <div className="action-bar">
        <button className="btn btn-primary" onClick={startCamera} disabled={streaming}>
          🎥 Start Camera
        </button>
        <button className="btn btn-soft" onClick={stopCamera} disabled={!streaming}>
          ⏹ Stop Camera
        </button>
        <button className="btn btn-soft" onClick={testTelegram}>
          ✉️ Test Telegram
        </button>
        <button className="btn btn-danger" onClick={clearData}>
          🧹 Clear Data
        </button>
      </div>

      <div className="status-strip">
        <span className={`status-pill ${statusClass}`}>
          {loading ? "⏳ Processing..." : `📍 Status: ${status}`}
        </span>
      </div>

      <div className="camera-section">
        <div className="camera-card main-camera-card">
          <h2>📷 Live Camera</h2>
          <div className="camera-frame">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} style={{ display: "none" }} />
          </div>
        </div>
      </div>

      <div className="two-grid">
        <div className="card elegant-card">
          <h2>🖼 Processed Output</h2>
          {processedImage ? (
            <img src={processedImage} alt="Processed detection output" className="preview-image" />
          ) : (
            <div className="empty-box">
              <span>🌿</span>
              <p>No processed frame yet.</p>
            </div>
          )}
        </div>

        <div className="card elegant-card">
          <h2>📝 Recent Events</h2>
          {events.length === 0 ? (
            <div className="empty-box">
              <span>🫧</span>
              <p>No events logged yet.</p>
            </div>
          ) : (
            <div className="events-list">
              {events.map((event) => (
                <div className="event-item" key={event.id}>
                  <div className="event-top">
                    <strong>{event.status}</strong>
                    <span>{event.created_at}</span>
                  </div>
                  <div className="event-bottom">
                    <span>📍 {event.source}</span>
                    <span>{event.extra_text}</span>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}