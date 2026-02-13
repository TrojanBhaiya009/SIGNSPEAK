import { useState, useRef, useEffect, useCallback } from 'react'
import { useUser, useAuth, UserButton, SignIn } from '@clerk/clerk-react'
import { io } from 'socket.io-client'
import './App.css'

const BACKEND_URL = import.meta.env.VITE_BACKEND_URL || 'http://localhost:5000'
const ICE_SERVERS = {
  iceServers: [
    { urls: 'stun:stun.l.google.com:19302' },
    { urls: 'stun:stun1.l.google.com:19302' },
    { urls: 'stun:stun2.l.google.com:19302' },
  ],
}

function App() {
  const { user, isLoaded } = useUser()
  const { isSignedIn } = useAuth()

  // ── Refs ──
  const localVideoRef = useRef(null)
  const previewVideoRef = useRef(null)
  const socketRef = useRef(null)
  const localStreamRef = useRef(null)
  const screenStreamRef = useRef(null)
  const peersRef = useRef({})
  const handsRef = useRef(null)
  const aslFrameRef = useRef(null)
  const chatEndRef = useRef(null)

  // ── Auth ──
  const [username, setUsername] = useState('')
  const [isSettingUsername, setIsSettingUsername] = useState(false)

  // ── Screens ──
  const [screen, setScreen] = useState('home')  // home | preview | call
  const [backendStatus, setBackendStatus] = useState('checking')
  const [toast, setToast] = useState('')

  // ── Meeting ──
  const [meetingCode, setMeetingCode] = useState('')
  const [joinCode, setJoinCode] = useState('')
  const [isHost, setIsHost] = useState(false)
  const [participants, setParticipants] = useState([])
  const [pendingParticipants, setPendingParticipants] = useState([])
  const [waitingApproval, setWaitingApproval] = useState(false)
  const [meetingMode, setMeetingMode] = useState('create') // create | join
  const [meetingTime, setMeetingTime] = useState(0)

  // ── Media ──
  const [isMuted, setIsMuted] = useState(false)
  const [isVideoOff, setIsVideoOff] = useState(false)
  const [isScreenSharing, setIsScreenSharing] = useState(false)

  // ── Remote streams ──
  const [remoteStreams, setRemoteStreams] = useState({})

  // ── Track when stream is ready ──
  const [streamReady, setStreamReady] = useState(0)

  // ── Panels ──
  const [showChat, setShowChat] = useState(false)
  const [showParticipants, setShowParticipants] = useState(false)
  const [showInfo, setShowInfo] = useState(false)

  // ── Chat ──
  const [chatMessages, setChatMessages] = useState([])
  const [chatInput, setChatInput] = useState('')
  const [unreadCount, setUnreadCount] = useState(0)

  // ── ASL ──
  const [detected, setDetected] = useState('')
  const [confidence, setConfidence] = useState(0)
  const [caption, setCaption] = useState('')
  const [showCaptions, setShowCaptions] = useState(true)

  // ── Socket ID ──
  const [mySocketId, setMySocketId] = useState('')

  // ── Helpers ──
  const flash = useCallback((msg) => {
    setToast(msg)
    setTimeout(() => setToast(''), 3000)
  }, [])

  const generateCode = () => 'SL-' + Math.random().toString(36).substring(2, 11).toUpperCase()

  const copyMeetingLink = () => {
    const link = `${window.location.origin}?code=${meetingCode}`
    navigator.clipboard.writeText(link)
    flash('Meeting link copied to clipboard')
  }

  const copyMeetingCode = () => {
    navigator.clipboard.writeText(meetingCode)
    flash('Meeting code copied')
  }

  // ── Timer ──
  useEffect(() => {
    if (screen !== 'call') { setMeetingTime(0); return }
    const iv = setInterval(() => setMeetingTime((t) => t + 1), 1000)
    return () => clearInterval(iv)
  }, [screen])

  const formatTime = (s) => {
    const m = Math.floor(s / 60)
    const sec = s % 60
    return `${m}:${sec.toString().padStart(2, '0')}`
  }

  // =================================================================
  // WebRTC helpers
  // =================================================================

  const createPeerConnectionRef = useRef(null)
  const handleOfferRef = useRef(null)
  const removePeerRef = useRef(null)

  const removePeer = useCallback((sid) => {
    const peer = peersRef.current[sid]
    if (peer?.pc) peer.pc.close()
    delete peersRef.current[sid]
    setRemoteStreams((prev) => { const n = { ...prev }; delete n[sid]; return n })
  }, [])

  const cleanUpAllPeers = useCallback(() => {
    Object.keys(peersRef.current).forEach(removePeer)
  }, [removePeer])

  const createPeerConnection = useCallback(async (remoteSid, remoteUsername, shouldCreateOffer) => {
    if (peersRef.current[remoteSid]) return
    const pc = new RTCPeerConnection(ICE_SERVERS)
    if (localStreamRef.current) {
      localStreamRef.current.getTracks().forEach((track) => pc.addTrack(track, localStreamRef.current))
    }

    pc.onicecandidate = (e) => {
      if (e.candidate) socketRef.current?.emit('ice-candidate', { target: remoteSid, candidate: e.candidate })
    }

    pc.ontrack = (e) => {
      const [stream] = e.streams
      if (!stream) return
      peersRef.current[remoteSid] = { ...peersRef.current[remoteSid], stream }
      setRemoteStreams((prev) => ({ ...prev, [remoteSid]: { stream, username: remoteUsername, audio: true, video: true } }))
    }

    pc.onconnectionstatechange = () => {
      if (pc.connectionState === 'failed' || pc.connectionState === 'closed') removePeer(remoteSid)
    }

    peersRef.current[remoteSid] = { pc, stream: null, username: remoteUsername, audio: true, video: true }

    if (shouldCreateOffer) {
      const offer = await pc.createOffer()
      await pc.setLocalDescription(offer)
      socketRef.current?.emit('webrtc-offer', { target: remoteSid, offer: pc.localDescription })
    }
  }, [removePeer])

  const handleOffer = useCallback(async (data) => {
    const { from_sid, from_username, offer } = data
    if (!peersRef.current[from_sid]) await createPeerConnectionRef.current(from_sid, from_username, false)
    const peer = peersRef.current[from_sid]
    if (!peer?.pc) return
    await peer.pc.setRemoteDescription(new RTCSessionDescription(offer))
    const answer = await peer.pc.createAnswer()
    await peer.pc.setLocalDescription(answer)
    socketRef.current?.emit('webrtc-answer', { target: from_sid, answer: peer.pc.localDescription })
  }, [])

  useEffect(() => {
    createPeerConnectionRef.current = createPeerConnection
    handleOfferRef.current = handleOffer
    removePeerRef.current = removePeer
  }, [createPeerConnection, handleOffer, removePeer])

  // =================================================================
  // Socket.IO
  // =================================================================

  useEffect(() => {
    const socket = io(BACKEND_URL, { transports: ['websocket', 'polling'], reconnection: true, reconnectionAttempts: 10, reconnectionDelay: 1000 })
    socketRef.current = socket

    socket.on('connect', () => { setBackendStatus('connected'); setMySocketId(socket.id) })
    socket.on('disconnect', () => setBackendStatus('disconnected'))

    socket.on('meeting-created', (d) => { if (d.success) { setParticipants(d.meeting.participants); setPendingParticipants(d.meeting.pending || []) } })
    socket.on('waiting-approval', () => setWaitingApproval(true))
    socket.on('admitted', (d) => { setWaitingApproval(false); setParticipants(d.participants || []); flash('You have been admitted') })
    socket.on('rejected', () => { setWaitingApproval(false); flash('Join request denied'); setScreen('home') })
    socket.on('join-error', (d) => { flash(d.error || 'Failed to join'); setWaitingApproval(false) })
    socket.on('pending-participant', (d) => { setPendingParticipants(d.pending || []); flash(`${d.participant.username} wants to join`) })
    socket.on('participants-updated', (d) => { setParticipants(d.participants || []); setPendingParticipants(d.pending || []) })

    socket.on('new-peer', async (d) => { if (d.should_create_offer) await createPeerConnectionRef.current?.(d.sid, d.username, true) })
    socket.on('webrtc-offer', async (d) => { await handleOfferRef.current?.(d) })
    socket.on('webrtc-answer', async (d) => { const p = peersRef.current[d.from_sid]; if (p?.pc) await p.pc.setRemoteDescription(new RTCSessionDescription(d.answer)) })
    socket.on('ice-candidate', async (d) => { const p = peersRef.current[d.from_sid]; if (p?.pc && d.candidate) { try { await p.pc.addIceCandidate(new RTCIceCandidate(d.candidate)) } catch { /* */ } } })
    socket.on('peer-disconnected', (d) => { removePeerRef.current?.(d.sid) })

    socket.on('peer-media-state', (d) => {
      setRemoteStreams((prev) => { const ex = prev[d.sid]; if (!ex) return prev; return { ...prev, [d.sid]: { ...ex, audio: d.audio, video: d.video } } })
      if (peersRef.current[d.sid]) { peersRef.current[d.sid].audio = d.audio; peersRef.current[d.sid].video = d.video }
    })

    socket.on('chat-message', (msg) => {
      setChatMessages((prev) => [...prev, msg])
      // increment unread if chat panel is closed
      setUnreadCount((prev) => prev + 1)
    })

    socket.on('prediction', (d) => {
      if (d.sign) setDetected(d.sign)
      setConfidence(Math.round((d.confidence || 0) * 100))
      if (d.caption !== undefined) setCaption(d.caption)
    })

    return () => { socket.disconnect() }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Username from Clerk ──
  const userIdRef = useRef(null)
  useEffect(() => {
    if (!isLoaded || !user) return
    if (userIdRef.current === user.id) return
    userIdRef.current = user.id
    setUsername(user.username || user.firstName || 'User')
    setIsSettingUsername(!user.username)
  }, [isLoaded, user?.id]) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Health check ──
  useEffect(() => {
    if (!isLoaded || !isSignedIn) return
    const check = async () => { try { const r = await fetch(`${BACKEND_URL}/health`); setBackendStatus(r.ok ? 'connected' : 'error') } catch { setBackendStatus('disconnected') } }
    check()
    const iv = setInterval(check, 15000)
    return () => clearInterval(iv)
  }, [isLoaded, isSignedIn])

  // ── Attach local stream to video elements when screens mount or stream changes ──
  useEffect(() => {
    if (screen === 'preview' && localStreamRef.current && previewVideoRef.current) {
      previewVideoRef.current.srcObject = localStreamRef.current
    }
    if (screen === 'call' && localStreamRef.current && localVideoRef.current) {
      localVideoRef.current.srcObject = localStreamRef.current
    }
  }, [screen, streamReady])

  // ── Chat auto-scroll ──
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [chatMessages])

  // ── Reset unread when opening chat ──
  useEffect(() => { if (showChat) setUnreadCount(0) }, [showChat])

  // =================================================================
  // Media
  // =================================================================

  const getLocalStream = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true })
    localStreamRef.current = stream
    if (localVideoRef.current) localVideoRef.current.srcObject = stream
    return stream
  }

  const toggleMute = () => {
    const next = !isMuted; setIsMuted(next)
    localStreamRef.current?.getAudioTracks().forEach((t) => { t.enabled = !next })
    socketRef.current?.emit('media-state', { code: meetingCode, audio: !next, video: !isVideoOff })
  }

  const toggleVideo = () => {
    const next = !isVideoOff; setIsVideoOff(next)
    localStreamRef.current?.getVideoTracks().forEach((t) => { t.enabled = !next })
    socketRef.current?.emit('media-state', { code: meetingCode, audio: !isMuted, video: !next })
  }

  const toggleScreenShare = async () => {
    if (isScreenSharing) {
      screenStreamRef.current?.getTracks().forEach((t) => t.stop())
      screenStreamRef.current = null
      const camTrack = localStreamRef.current?.getVideoTracks()[0]
      if (camTrack) Object.values(peersRef.current).forEach(({ pc }) => { const s = pc.getSenders().find((s) => s.track?.kind === 'video'); if (s) s.replaceTrack(camTrack) })
      setIsScreenSharing(false)
      socketRef.current?.emit('screen-share-stopped', { code: meetingCode })
    } else {
      try {
        const ss = await navigator.mediaDevices.getDisplayMedia({ video: true })
        screenStreamRef.current = ss
        const st = ss.getVideoTracks()[0]
        Object.values(peersRef.current).forEach(({ pc }) => { const s = pc.getSenders().find((s) => s.track?.kind === 'video'); if (s) s.replaceTrack(st) })
        st.onended = () => toggleScreenShare()
        setIsScreenSharing(true)
        socketRef.current?.emit('screen-share-started', { code: meetingCode })
      } catch { /* cancel */ }
    }
  }

  // =================================================================
  // ASL detection
  // =================================================================

  const initASLDetection = () => {
    if (!window.Hands) return
    const hands = new window.Hands({ locateFile: (f) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${f}` })
    hands.setOptions({ maxNumHands: 1, modelComplexity: 0, minDetectionConfidence: 0.5, minTrackingConfidence: 0.4 })

    hands.onResults((results) => {
      if (results.multiHandLandmarks?.length > 0) {
        const lm = results.multiHandLandmarks[0]
        const flat = lm.flatMap((l) => [l.x, l.y, l.z])
        const wx = flat[0], wy = flat[1], wz = flat[2]
        const pts = []
        for (let i = 0; i < 21; i++) pts.push(flat[i * 3] - wx, flat[i * 3 + 1] - wy, flat[i * 3 + 2] - wz)
        let maxDist = 0
        for (let i = 0; i < 21; i++) { const d = Math.sqrt(pts[i * 3] ** 2 + pts[i * 3 + 1] ** 2 + pts[i * 3 + 2] ** 2); if (d > maxDist) maxDist = d }
        const normed = maxDist > 0 ? pts.map((v) => v / maxDist) : pts
        socketRef.current?.emit('predict-landmarks', { landmarks: normed })
      }
    })

    handsRef.current = hands
    let processing = false
    const processFrame = async () => {
      const vid = localVideoRef.current
      if (!vid || !vid.srcObject || !handsRef.current) { aslFrameRef.current = requestAnimationFrame(processFrame); return }
      if (vid.readyState >= 2 && !processing) { processing = true; try { await handsRef.current.send({ image: vid }) } catch { /* */ } processing = false }
      aslFrameRef.current = requestAnimationFrame(processFrame)
    }
    aslFrameRef.current = requestAnimationFrame(processFrame)
  }

  // =================================================================
  // Caption controls
  // =================================================================

  const clearCaption = async () => { try { await fetch(`${BACKEND_URL}/caption/clear`, { method: 'POST' }); setCaption('') } catch { /* */ } }
  const addSpace = async () => { try { const r = await fetch(`${BACKEND_URL}/caption/space`, { method: 'POST' }); const d = await r.json(); if (d.caption !== undefined) setCaption(d.caption) } catch { /* */ } }
  const doBackspace = async () => { try { const r = await fetch(`${BACKEND_URL}/caption/backspace`, { method: 'POST' }); const d = await r.json(); if (d.caption !== undefined) setCaption(d.caption) } catch { /* */ } }

  // =================================================================
  // Meeting actions
  // =================================================================

  const goToPreview = async (mode) => {
    if (mode === 'create') setMeetingCode(generateCode())
    setMeetingMode(mode)
    setScreen('preview')
    // Start camera for preview
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1280 }, height: { ideal: 720 } }, audio: true })
      localStreamRef.current = stream
      setStreamReady((c) => c + 1) // trigger re-attach
    } catch (e) { flash('Camera error: ' + e.message) }
  }

  const startMeeting = async () => {
    try {
      const code = meetingCode || generateCode()
      setMeetingCode(code)
      // Reuse preview stream if available, otherwise get new one
      if (!localStreamRef.current) await getLocalStream()
      if (isMuted) localStreamRef.current?.getAudioTracks().forEach((t) => { t.enabled = false })
      if (isVideoOff) localStreamRef.current?.getVideoTracks().forEach((t) => { t.enabled = false })
      setIsHost(true)
      socketRef.current?.emit('create-meeting', { code, user_id: user.id, username })
      setScreen('call')
      initASLDetection()
    } catch (e) { flash('Camera error: ' + e.message) }
  }

  const joinMeeting = async () => {
    const code = (meetingMode === 'join' ? joinCode : meetingCode).trim()
    if (!code) { flash('Enter a meeting code'); return }
    try {
      setMeetingCode(code)
      // Reuse preview stream if available, otherwise get new one
      if (!localStreamRef.current) await getLocalStream()
      if (isMuted) localStreamRef.current?.getAudioTracks().forEach((t) => { t.enabled = false })
      if (isVideoOff) localStreamRef.current?.getVideoTracks().forEach((t) => { t.enabled = false })
      setIsHost(false)
      socketRef.current?.emit('join-meeting', { code, user_id: user.id, username })
      setScreen('call')
      initASLDetection()
    } catch (e) { flash('Camera error: ' + e.message) }
  }

  const endCall = () => {
    localStreamRef.current?.getTracks().forEach((t) => t.stop()); localStreamRef.current = null
    screenStreamRef.current?.getTracks().forEach((t) => t.stop()); screenStreamRef.current = null
    if (aslFrameRef.current) { cancelAnimationFrame(aslFrameRef.current); aslFrameRef.current = null }
    cleanUpAllPeers()
    socketRef.current?.emit('leave-meeting', { code: meetingCode })
    setScreen('home'); setIsHost(false); setParticipants([]); setPendingParticipants([]); setWaitingApproval(false)
    setRemoteStreams({}); setChatMessages([]); setCaption(''); setDetected(''); setConfidence(0); setIsScreenSharing(false)
    setIsMuted(false); setIsVideoOff(false); setShowChat(false); setShowParticipants(false); setShowInfo(false)
  }

  const admitParticipant = (sid) => socketRef.current?.emit('admit-participant', { code: meetingCode, sid })
  const rejectParticipant = (sid) => socketRef.current?.emit('reject-participant', { code: meetingCode, sid })

  const sendChat = () => {
    const text = chatInput.trim()
    if (!text) return
    socketRef.current?.emit('chat-message', { code: meetingCode, text, type: 'text' })
    setChatInput('')
  }

  // =================================================================
  // Video grid layout
  // =================================================================

  const remoteEntries = Object.entries(remoteStreams)
  const totalVideos = 1 + remoteEntries.length
  const gridClass = totalVideos <= 1 ? 'g-1' : totalVideos <= 2 ? 'g-2' : totalVideos <= 4 ? 'g-4' : totalVideos <= 6 ? 'g-6' : 'g-many'

  // Right panel open?
  const sidebarOpen = showChat || showParticipants || showInfo

  // =================================================================
  // Render
  // =================================================================

  if (!isLoaded) return (
    <div className="gm-app">
      <div className="gm-loader"><div className="gm-spinner" /><p>Loading SignSpeak...</p></div>
    </div>
  )

  if (!isSignedIn) return (
    <div className="gm-app">
      <div className="gm-login">
        <div className="gm-login-card">
          <div className="gm-login-logo">🤟</div>
          <h1>SignSpeak</h1>
          <p>Real-time ASL recognition in video calls</p>
          <SignIn appearance={{ elements: { rootBox: { width: '100%', maxWidth: '400px' }, card: { background: '#fff', borderRadius: '16px', border: 'none' } } }} />
        </div>
      </div>
    </div>
  )

  if (isSettingUsername) return (
    <div className="gm-app">
      <div className="gm-login">
        <div className="gm-login-card">
          <h2>Welcome! 👋</h2>
          <p style={{ marginBottom: 24 }}>Choose a display name</p>
          <input className="gm-input" placeholder="Your name" value={username} onChange={(e) => setUsername(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && username.trim() && setIsSettingUsername(false)} />
          <button className="gm-btn gm-btn-primary" style={{ marginTop: 16, width: '100%' }} onClick={() => { if (username.trim()) setIsSettingUsername(false) }}>Continue</button>
        </div>
      </div>
    </div>
  )

  // ────────────────────────────────────────────────────────
  // HOME SCREEN (Google Meet landing)
  // ────────────────────────────────────────────────────────

  if (screen === 'home') return (
    <div className="gm-app">
      {toast && <div className="gm-toast">{toast}</div>}
      {/* Top bar */}
      <header className="gm-topbar">
        <div className="gm-topbar-brand">
          <span className="gm-logo">🤟</span>
          <span className="gm-brand-name">SignSpeak</span>
        </div>
        <div className="gm-topbar-time">{new Date().toLocaleDateString('en-US', { weekday: 'short', month: 'short', day: 'numeric' })} · {new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</div>
        <div className="gm-topbar-actions">
          <div className={`gm-status-dot ${backendStatus === 'connected' ? 'on' : 'off'}`} title={`AI: ${backendStatus}`} />
          <UserButton />
        </div>
      </header>

      {/* Main content */}
      <main className="gm-home">
        <div className="gm-home-left">
          <h1 className="gm-home-title">Video calls with <span className="gm-highlight">sign language AI</span></h1>
          <p className="gm-home-subtitle">SignSpeak uses real-time ASL recognition to make video meetings accessible to everyone.</p>

          <div className="gm-home-actions">
            <button className="gm-btn gm-btn-primary gm-btn-new" onClick={() => goToPreview('create')}>
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              New meeting
            </button>

            <div className="gm-join-input-wrap">
              <svg className="gm-join-icon" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="3" y="11" width="18" height="11" rx="2" /><path d="M7 11V7a5 5 0 0110 0v4" /></svg>
              <input className="gm-join-input" placeholder="Enter a code or link" value={joinCode} onChange={(e) => setJoinCode(e.target.value.toUpperCase())} onKeyDown={(e) => { if (e.key === 'Enter' && joinCode.trim()) goToPreview('join') }} />
              {joinCode.trim() && <button className="gm-btn gm-btn-text" onClick={() => goToPreview('join')}>Join</button>}
            </div>
          </div>
        </div>

        <div className="gm-home-right">
          <div className="gm-illustration">
            <div className="gm-illust-card">
              <div className="gm-illust-icon">🤟</div>
              <p>Start or join a meeting to begin using ASL recognition</p>
            </div>
          </div>
        </div>
      </main>

      {/* Feature cards */}
      <section className="gm-features">
        {[
          ['🎥', 'HD Video', 'Crystal-clear peer-to-peer video with WebRTC'],
          ['🤖', 'ASL Detection', 'Real-time sign language recognition powered by ML'],
          ['💬', 'Live Captions', 'Auto-generated captions from detected signs'],
          ['🖥️', 'Screen Share', 'Share your screen with one click'],
          ['👥', 'Multi-Party', 'Support for multiple participants in a call'],
          ['🔒', 'Secure', 'End-to-end encrypted with unique meeting codes'],
        ].map(([icon, title, desc]) => (
          <div className="gm-feat-card" key={title}>
            <span className="gm-feat-icon">{icon}</span>
            <strong>{title}</strong>
            <span className="gm-feat-desc">{desc}</span>
          </div>
        ))}
      </section>
    </div>
  )

  // ────────────────────────────────────────────────────────
  // PREVIEW SCREEN (before joining)
  // ────────────────────────────────────────────────────────

  if (screen === 'preview') return (
    <div className="gm-app">
      {toast && <div className="gm-toast">{toast}</div>}
      <header className="gm-topbar">
        <div className="gm-topbar-brand" onClick={() => setScreen('home')} style={{ cursor: 'pointer' }}>
          <span className="gm-logo">🤟</span><span className="gm-brand-name">SignSpeak</span>
        </div>
        <div className="gm-topbar-actions"><UserButton /></div>
      </header>

      <div className="gm-preview">
        <div className="gm-preview-video-wrap">
          <div className="gm-preview-video-box">
            <video ref={previewVideoRef} autoPlay playsInline muted className={`gm-preview-vid ${isVideoOff ? 'hidden' : ''}`} />
            {isVideoOff && <div className="gm-preview-placeholder"><span className="gm-avatar-lg">{username?.[0]?.toUpperCase() || '?'}</span></div>}
          </div>
          <div className="gm-preview-controls">
            <button className={`gm-ctrl-btn ${isMuted ? 'off' : ''}`} onClick={() => { const next = !isMuted; setIsMuted(next); localStreamRef.current?.getAudioTracks().forEach((t) => { t.enabled = !next }) }} title={isMuted ? 'Unmute' : 'Mute'}>
              {isMuted
                ? <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M1.5 4.5l21 15m-21 0l21-15M12 1a3 3 0 00-3 3v4.5M15 9.34V4a3 3 0 00-5.94-.6M9 9v3a3 3 0 005.12 2.12M19 10v2a7 7 0 01-.11 1.23M5 10v2a7 7 0 0011.47 5.38M12 19v3m-4 0h8" /></svg>
                : <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" /><path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" /></svg>
              }
            </button>
            <button className={`gm-ctrl-btn ${isVideoOff ? 'off' : ''}`} onClick={() => { const next = !isVideoOff; setIsVideoOff(next); localStreamRef.current?.getVideoTracks().forEach((t) => { t.enabled = !next }) }} title={isVideoOff ? 'Turn on camera' : 'Turn off camera'}>
              {isVideoOff
                ? <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M1 1l22 22M17 17H5a2 2 0 01-2-2V7a2 2 0 012-2h2m10 0h1a2 2 0 012 2v4l4-2.5v7" /></svg>
                : <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              }
            </button>
          </div>
        </div>

        <div className="gm-preview-join">
          <h2>{meetingMode === 'create' ? 'Ready to start?' : 'Ready to join?'}</h2>
          {meetingMode === 'join' && (
            <div className="gm-preview-code">
              <label>Meeting code</label>
              <input className="gm-input" value={joinCode} onChange={(e) => setJoinCode(e.target.value.toUpperCase())} placeholder="SL-XXXXXXXXX" />
            </div>
          )}
          {meetingMode === 'create' && (
            <div className="gm-preview-code">
              <label>Your meeting code</label>
              <div className="gm-code-row">
                <span className="gm-code-display">{meetingCode}</span>
                <button className="gm-btn gm-btn-icon" onClick={copyMeetingCode} title="Copy code">
                  <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>
                </button>
              </div>
            </div>
          )}
          <p className="gm-preview-info">No one else is here yet</p>
          <button className="gm-btn gm-btn-primary gm-btn-lg" onClick={meetingMode === 'create' ? startMeeting : joinMeeting}>
            {meetingMode === 'create' ? 'Start meeting' : 'Join now'}
          </button>
          <button className="gm-btn gm-btn-text" onClick={() => { localStreamRef.current?.getTracks().forEach((t) => t.stop()); localStreamRef.current = null; setScreen('home') }}>← Back</button>
        </div>
      </div>
    </div>
  )

  // ────────────────────────────────────────────────────────
  // CALL SCREEN (Google Meet in-call)
  // ────────────────────────────────────────────────────────

  return (
    <div className="gm-app gm-in-call">
      {toast && <div className="gm-toast">{toast}</div>}

      {/* Waiting overlay */}
      {waitingApproval && (
        <div className="gm-waiting-overlay">
          <div className="gm-waiting-card">
            <div className="gm-spinner" />
            <h3>Asking to be let in...</h3>
            <p>The host will let you in soon</p>
            <button className="gm-btn gm-btn-text" onClick={endCall}>Cancel</button>
          </div>
        </div>
      )}

      {/* Video area */}
      <div className={`gm-call-main ${sidebarOpen ? 'with-panel' : ''}`}>
        <div className={`gm-video-grid ${gridClass}`}>
          {/* Local tile */}
          <div className={`gm-tile ${totalVideos === 1 ? 'gm-tile-solo' : ''}`}>
            <video ref={localVideoRef} autoPlay playsInline muted className={`gm-tile-video ${isVideoOff ? 'hidden' : ''}`} />
            {isVideoOff && <div className="gm-tile-avatar"><span className="gm-avatar">{username?.[0]?.toUpperCase() || '?'}</span></div>}
            <div className="gm-tile-bar">
              <span className="gm-tile-name">{username} (You)</span>
              {isMuted && <span className="gm-tile-muted">🔇</span>}
            </div>
            {isHost && <span className="gm-host-tag">Host</span>}
          </div>

          {/* Remote tiles */}
          {remoteEntries.map(([sid, remote]) => (
            <RemoteVideo key={sid} remote={remote} />
          ))}
        </div>

        {/* ASL Detection badge */}
        {detected && (
          <div className="gm-detect-badge">
            <span className="gm-detect-letter">{detected}</span>
            <div className="gm-detect-bar"><div className="gm-detect-fill" style={{ width: confidence + '%' }} /></div>
            <span className="gm-detect-conf">{confidence}%</span>
          </div>
        )}

        {/* Caption */}
        {showCaptions && caption && (
          <div className="gm-caption-bar">
            <span className="gm-caption-text">{caption}</span>
            <div className="gm-caption-actions">
              <button onClick={doBackspace} title="Backspace">⌫</button>
              <button onClick={addSpace} title="Space">_</button>
              <button onClick={clearCaption} title="Clear">✕</button>
            </div>
          </div>
        )}

        {/* Bottom control bar */}
        <div className="gm-controls">
          <div className="gm-controls-left">
            <span className="gm-meeting-id" onClick={copyMeetingCode} title="Click to copy">{meetingCode}</span>
            <span className="gm-meeting-timer">{formatTime(meetingTime)}</span>
          </div>

          <div className="gm-controls-center">
            <button className={`gm-ctrl-btn ${isMuted ? 'off' : ''}`} onClick={toggleMute} title={isMuted ? 'Unmute' : 'Mute'}>
              {isMuted
                ? <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M1.5 4.5l21 15m-21 0l21-15M12 1a3 3 0 00-3 3v4.5M15 9.34V4a3 3 0 00-5.94-.6M9 9v3a3 3 0 005.12 2.12M19 10v2a7 7 0 01-.11 1.23M5 10v2a7 7 0 0011.47 5.38M12 19v3m-4 0h8" /></svg>
                : <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 1a3 3 0 00-3 3v8a3 3 0 006 0V4a3 3 0 00-3-3z" /><path d="M19 10v2a7 7 0 01-14 0v-2M12 19v4M8 23h8" /></svg>
              }
            </button>
            <button className={`gm-ctrl-btn ${isVideoOff ? 'off' : ''}`} onClick={toggleVideo} title={isVideoOff ? 'Turn on camera' : 'Turn off camera'}>
              {isVideoOff
                ? <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M1 1l22 22M17 17H5a2 2 0 01-2-2V7a2 2 0 012-2h2m10 0h1a2 2 0 012 2v4l4-2.5v7" /></svg>
                : <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M15 10l4.553-2.276A1 1 0 0121 8.618v6.764a1 1 0 01-1.447.894L15 14M5 18h8a2 2 0 002-2V8a2 2 0 00-2-2H5a2 2 0 00-2 2v8a2 2 0 002 2z" /></svg>
              }
            </button>
            <button className={`gm-ctrl-btn ${isScreenSharing ? 'sharing' : ''}`} onClick={toggleScreenShare} title={isScreenSharing ? 'Stop presenting' : 'Present now'}>
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="3" width="20" height="14" rx="2" /><path d="M8 21h8M12 17v4" /></svg>
            </button>
            <button className={`gm-ctrl-btn ${showCaptions ? 'active' : ''}`} onClick={() => setShowCaptions(!showCaptions)} title="Toggle captions">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="2" y="4" width="20" height="16" rx="2" /><path d="M7 12h2m4 0h4M7 16h10" /></svg>
            </button>
            <button className="gm-ctrl-btn gm-end-btn" onClick={endCall} title="Leave call">
              <svg width="24" height="24" viewBox="0 0 24 24" fill="currentColor"><path d="M12 9c-1.6 0-3.15.25-4.6.72v3.1c0 .39-.23.74-.56.9-.98.49-1.87 1.12-2.66 1.85-.18.18-.43.28-.7.28-.28 0-.53-.11-.71-.29L.29 13.08a.956.956 0 010-1.36C3.69 8.68 7.62 7 12 7s8.31 1.68 11.71 4.72c.37.37.37.98 0 1.36l-2.48 2.48c-.18.18-.43.29-.71.29-.27 0-.52-.1-.7-.28-.79-.73-1.68-1.36-2.66-1.85a.994.994 0 01-.56-.9v-3.1C15.15 9.25 13.6 9 12 9z" /></svg>
            </button>
          </div>

          <div className="gm-controls-right">
            <button className={`gm-ctrl-btn-sm ${showInfo ? 'active' : ''}`} onClick={() => { setShowInfo(!showInfo); setShowChat(false); setShowParticipants(false) }} title="Meeting details">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4M12 8h.01" /></svg>
            </button>
            <button className={`gm-ctrl-btn-sm ${showParticipants ? 'active' : ''}`} onClick={() => { setShowParticipants(!showParticipants); setShowChat(false); setShowInfo(false) }} title="People">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M17 21v-2a4 4 0 00-4-4H5a4 4 0 00-4 4v2" /><circle cx="9" cy="7" r="4" /><path d="M23 21v-2a4 4 0 00-3-3.87M16 3.13a4 4 0 010 7.75" /></svg>
              {participants.length > 0 && <span className="gm-badge-count">{participants.length}</span>}
            </button>
            <button className={`gm-ctrl-btn-sm ${showChat ? 'active' : ''}`} onClick={() => { setShowChat(!showChat); setShowParticipants(false); setShowInfo(false) }} title="Chat">
              <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z" /></svg>
              {unreadCount > 0 && !showChat && <span className="gm-badge-count">{unreadCount}</span>}
            </button>
          </div>
        </div>
      </div>

      {/* Right side panel */}
      {sidebarOpen && (
        <div className="gm-panel">
          {/* Meeting info panel */}
          {showInfo && (
            <>
              <div className="gm-panel-header">
                <h3>Meeting details</h3>
                <button className="gm-panel-close" onClick={() => setShowInfo(false)}>✕</button>
              </div>
              <div className="gm-panel-body">
                <div className="gm-info-section">
                  <label>Joining info</label>
                  <div className="gm-info-link">{window.location.origin}?code={meetingCode}</div>
                  <button className="gm-btn gm-btn-outline" onClick={copyMeetingLink}>
                    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><rect x="9" y="9" width="13" height="13" rx="2" /><path d="M5 15H4a2 2 0 01-2-2V4a2 2 0 012-2h9a2 2 0 012 2v1" /></svg>
                    Copy joining info
                  </button>
                </div>
                <div className="gm-info-section">
                  <label>Meeting code</label>
                  <div className="gm-code-inline">{meetingCode}</div>
                </div>
              </div>
            </>
          )}

          {/* Participants panel */}
          {showParticipants && (
            <>
              <div className="gm-panel-header">
                <h3>People ({participants.length})</h3>
                <button className="gm-panel-close" onClick={() => setShowParticipants(false)}>✕</button>
              </div>
              <div className="gm-panel-body">
                {/* Pending */}
                {isHost && pendingParticipants.length > 0 && (
                  <div className="gm-pending-section">
                    <h4>Waiting to join ({pendingParticipants.length})</h4>
                    {pendingParticipants.map((p) => (
                      <div key={p.sid} className="gm-person-row gm-pending-row">
                        <span className="gm-person-avatar">{p.username?.[0]?.toUpperCase()}</span>
                        <span className="gm-person-name">{p.username}</span>
                        <div className="gm-pending-btns">
                          <button className="gm-btn gm-btn-sm gm-btn-admit" onClick={() => admitParticipant(p.sid)}>Admit</button>
                          <button className="gm-btn gm-btn-sm gm-btn-deny" onClick={() => rejectParticipant(p.sid)}>Deny</button>
                        </div>
                      </div>
                    ))}
                  </div>
                )}
                {/* In call */}
                <div className="gm-people-list">
                  {participants.map((p) => (
                    <div key={p.sid} className="gm-person-row">
                      <span className="gm-person-avatar">{p.username?.[0]?.toUpperCase()}</span>
                      <span className="gm-person-name">{p.username} {p.is_host && <span className="gm-host-label">(Host)</span>}</span>
                      <div className="gm-person-icons">
                        {p.audio === false && <span title="Muted">🔇</span>}
                        {p.video === false && <span title="Camera off">📷</span>}
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </>
          )}

          {/* Chat panel */}
          {showChat && (
            <>
              <div className="gm-panel-header">
                <h3>In-call messages</h3>
                <button className="gm-panel-close" onClick={() => setShowChat(false)}>✕</button>
              </div>
              <div className="gm-panel-body gm-chat-body">
                <div className="gm-chat-messages">
                  {chatMessages.length === 0 && <div className="gm-chat-empty"><p>Messages can only be seen by people in the call and are deleted when the call ends.</p></div>}
                  {chatMessages.map((m) => (
                    <div key={m.id || m.timestamp} className={`gm-chat-msg ${m.sid === mySocketId ? 'mine' : ''}`}>
                      <span className="gm-chat-sender">{m.username}</span>
                      <span className="gm-chat-text">{m.text}</span>
                      <span className="gm-chat-time">{new Date(m.timestamp || Date.now()).toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' })}</span>
                    </div>
                  ))}
                  <div ref={chatEndRef} />
                </div>
                <div className="gm-chat-input-row">
                  <input className="gm-chat-input" placeholder="Send a message to everyone" value={chatInput} onChange={(e) => setChatInput(e.target.value)} onKeyDown={(e) => e.key === 'Enter' && sendChat()} />
                  <button className="gm-chat-send" onClick={sendChat} disabled={!chatInput.trim()}>
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor"><path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z" /></svg>
                  </button>
                </div>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

// =================================================================
// Remote video tile
// =================================================================

function RemoteVideo({ remote }) {
  const videoRef = useRef(null)
  useEffect(() => { if (videoRef.current && remote.stream) videoRef.current.srcObject = remote.stream }, [remote.stream])

  return (
    <div className="gm-tile">
      <video ref={videoRef} autoPlay playsInline className={`gm-tile-video ${remote.video === false ? 'hidden' : ''}`} />
      {remote.video === false && <div className="gm-tile-avatar"><span className="gm-avatar">{remote.username?.[0]?.toUpperCase() || '?'}</span></div>}
      <div className="gm-tile-bar">
        <span className="gm-tile-name">{remote.username}</span>
        {remote.audio === false && <span className="gm-tile-muted">🔇</span>}
      </div>
    </div>
  )
}

export default App
