import { useState, useRef, useEffect } from 'react'
import { useUser, useAuth, UserButton, SignIn } from '@clerk/clerk-react'
import './App.css'

function App() {
  const { user, isLoaded } = useUser()
  const { isSignedIn } = useAuth()
  const videoRef = useRef(null)
  const canvasRef = useRef(null)
  const handsRef = useRef(null)
  const cameraRef = useRef(null)
  
  // Auth states
  const [username, setUsername] = useState(user?.username || '')
  const [isSettingUsername, setIsSettingUsername] = useState(!user?.username)
  
  // Meeting states
  const [screen, setScreen] = useState('home')
  const [isActive, setIsActive] = useState(false)
  const [detected, setDetected] = useState('Waiting for hands...')
  const [confidence, setConfidence] = useState(0)
  const [transcript, setTranscript] = useState('')
  const [caption, setCaption] = useState('')  // Live caption from spelling
  const [backendStatus, setBackendStatus] = useState('Disconnected')
  const [meetingCode, setMeetingCode] = useState('SL-' + Math.random().toString(36).substr(2, 9).toUpperCase())
  const [joinCode, setJoinCode] = useState('')
  const [isMuted, setIsMuted] = useState(false)
  const [isVideoOff, setIsVideoOff] = useState(false)
  const [lastKeypoints, setLastKeypoints] = useState(null)
  
  // Participants management
  const [participants, setParticipants] = useState([])
  const [pendingParticipants, setPendingParticipants] = useState([])
  const [isHost, setIsHost] = useState(false)

  // Initialize app
  useEffect(() => {
    if (!isLoaded) return
    
    if (!isSignedIn) {
      setScreen('login')
      return
    }
    
    if (user && !participants.find(p => p.id === user.id)) {
      setParticipants([{
        id: user.id,
        name: username || user.username || 'Anonymous',
        joinedAt: new Date(),
        isHost: true
      }])
      setIsHost(true)
    }
    
    checkBackendStatus()
    const interval = setInterval(checkBackendStatus, 5000)
    return () => clearInterval(interval)
  }, [isLoaded, isSignedIn, user, username])

  // Recording loop - using CLIENT-SIDE MediaPipe for INSTANT detection
  useEffect(() => {
    if (screen === 'call' && isActive && videoRef.current && canvasRef.current) {
      initClientSideDetection()
    }
    return () => {
      if (cameraRef.current) {
        cameraRef.current.stop()
        cameraRef.current = null
      }
    }
  }, [screen, isActive])

  // Render loop for hand keypoints
  useEffect(() => {
    if (!isActive || screen !== 'call') return

    let rafId
    const renderLoop = () => {
      const canvas = canvasRef.current
      const video = videoRef.current

      if (canvas && video && video.readyState === 4) {
        const ctx = canvas.getContext('2d')
        
        // Draw video without flip
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height)
        
        if (lastKeypoints && lastKeypoints.length > 0) {
          drawHandKeypoints(ctx, lastKeypoints)
        }
      }
      
      rafId = requestAnimationFrame(renderLoop)
    }
    
    rafId = requestAnimationFrame(renderLoop)
    return () => cancelAnimationFrame(rafId)
  }, [isActive, screen, lastKeypoints])

  const checkBackendStatus = async () => {
    try {
      const res = await fetch('http://localhost:5000/health')
      if (res.ok) setBackendStatus('Connected')
      else setBackendStatus('Error')
    } catch {
      setBackendStatus('Disconnected')
    }
  }

  // CLIENT-SIDE MediaPipe detection - INSTANT, no network latency
  const lastDetectionRef = useRef({ count: 0, lastState: 'waiting' })
  
  const initClientSideDetection = async () => {
    if (!window.Hands || !window.Camera) {
      console.log('MediaPipe not loaded yet, falling back to server')
      recordingLoop()
      return
    }

    const hands = new window.Hands({
      locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/hands/${file}`
    })
    
    hands.setOptions({
      maxNumHands: 2,  // Detect both hands
      modelComplexity: 0,  // Lite model - fastest
      minDetectionConfidence: 0.5,
      minTrackingConfidence: 0.4
    })

    hands.onResults((results) => {
      const canvas = canvasRef.current
      const video = videoRef.current
      if (!canvas || !video) return

      const ctx = canvas.getContext('2d')
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height)

      const hasHands = results.multiHandLandmarks && results.multiHandLandmarks.length > 0
      
      // STRONG smoothing: slow to change, fast to update when stable
      if (hasHands) {
        // Quick ramp up when hands detected
        lastDetectionRef.current.count = Math.min(10, lastDetectionRef.current.count + 3)
      } else {
        // SLOW ramp down - prevent flickering
        lastDetectionRef.current.count = Math.max(0, lastDetectionRef.current.count - 1)
      }
      
      // Only change state with strong confidence
      const shouldShowDetected = lastDetectionRef.current.count >= 4
      const shouldShowWaiting = lastDetectionRef.current.count <= 1

      if (hasHands) {
        const keypoints = results.multiHandLandmarks.map(hand => 
          hand.map(p => [p.x * canvas.width, p.y * canvas.height])
        )
        setLastKeypoints(keypoints)
        drawHandKeypoints(ctx, keypoints)
        
        if (shouldShowDetected && lastDetectionRef.current.lastState !== 'detected') {
          const handCount = results.multiHandLandmarks.length
          setDetected(handCount === 2 ? '2 Hands detected' : 'Hand detected')
          setConfidence(85 + handCount * 5)
          lastDetectionRef.current.lastState = 'detected'
        }
        
        // Send to backend for letter prediction (async, non-blocking)
        sendImageToBackend()
      } else {
        setLastKeypoints(null)
        if (shouldShowWaiting && lastDetectionRef.current.lastState !== 'waiting') {
          setDetected('Waiting for hands...')
          setConfidence(0)
          lastDetectionRef.current.lastState = 'waiting'
        }
      }
    })

    handsRef.current = hands

    const camera = new window.Camera(videoRef.current, {
      onFrame: async () => {
        if (handsRef.current && videoRef.current) {
          await handsRef.current.send({ image: videoRef.current })
        }
      },
      width: 640,
      height: 480
    })
    
    cameraRef.current = camera
    camera.start()
    console.log('Client-side MediaPipe detection started - INSTANT mode')
  }

  // Send image to backend for ISL letter prediction
  const predictionInFlightRef = useRef(false)
  const sendImageToBackend = async () => {
    if (predictionInFlightRef.current || !videoRef.current) return
    
    predictionInFlightRef.current = true
    
    try {
      // Create small canvas for the hand region
      const tempCanvas = document.createElement('canvas')
      tempCanvas.width = 128
      tempCanvas.height = 128
      const ctx = tempCanvas.getContext('2d')
      ctx.drawImage(videoRef.current, 0, 0, 128, 128)
      
      const base64 = tempCanvas.toDataURL('image/jpeg', 0.7).split(',')[1]
      
      const res = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image: base64 })
      })
      const data = await res.json()
      
      if (data.sign && !data.sign.includes('Waiting') && !data.sign.includes('detected')) {
        setDetected(data.sign)
        setConfidence(Math.round((data.confidence || 0) * 100))
      }
      if (data.caption !== undefined) {
        setCaption(data.caption)
        // Always sync transcript with caption
        setTranscript(data.caption)
      }
    } catch (e) { 
      console.error('Backend prediction error:', e)
    }
    
    predictionInFlightRef.current = false
  }

  const startCall = async () => {
    try {
      const constraints = {
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      }
      if (!isMuted) constraints.audio = true

      const stream = await navigator.mediaDevices.getUserMedia(constraints)
      
      if (videoRef.current) {
        videoRef.current.srcObject = stream
        setIsActive(true)
        setTimeout(() => setScreen('call'), 100)
      }
    } catch (error) {
      console.error('Camera error:', error)
      alert('Cannot access camera:\n' + error.message)
    }
  }

  const endCall = () => {
    if (videoRef.current?.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop())
    }
    setIsActive(false)
    setScreen('home')
  }

  // Caption control functions
  const clearCaption = async () => {
    try {
      await fetch('http://localhost:5000/caption/clear', { method: 'POST' })
      setCaption('')
      setTranscript('')  // Also clear transcript
    } catch (e) { console.error(e) }
  }

  const addSpace = async () => {
    try {
      const res = await fetch('http://localhost:5000/caption/space', { method: 'POST' })
      const data = await res.json()
      if (data.caption !== undefined) setCaption(data.caption)
    } catch (e) { console.error(e) }
  }

  const backspace = async () => {
    try {
      const res = await fetch('http://localhost:5000/caption/backspace', { method: 'POST' })
      const data = await res.json()
      if (data.caption !== undefined) setCaption(data.caption)
    } catch (e) { console.error(e) }
  }

  // Track if a request is in flight to avoid piling up requests
  const requestInFlight = useRef(false)
  const frameCanvasRef = useRef(null)

  const recordingLoop = () => {
    const videoElement = videoRef.current
    const canvasElement = canvasRef.current

    if (isActive && videoElement && canvasElement && videoElement.readyState === 4) {
      // Skip if previous request still pending (non-blocking)
      if (requestInFlight.current) {
        setTimeout(recordingLoop, 16) // ~60fps check rate
        return
      }

      // Reuse canvas for performance - TINY for speed
      if (!frameCanvasRef.current) {
        frameCanvasRef.current = document.createElement('canvas')
        frameCanvasRef.current.width = 160  // Tiny for speed
        frameCanvasRef.current.height = 120
      }
      const tempCanvas = frameCanvasRef.current
      const tempCtx = tempCanvas.getContext('2d')
      tempCtx.drawImage(videoElement, 0, 0, tempCanvas.width, tempCanvas.height)

      requestInFlight.current = true

      // Use lower quality JPEG for faster transfer
      tempCanvas.toBlob(async (blob) => {
        if (!blob) {
          requestInFlight.current = false
          setTimeout(recordingLoop, 16)
          return
        }

        const base64 = await new Promise((resolve) => {
          const reader = new FileReader()
          reader.onloadend = () => resolve(reader.result.split(',')[1])
          reader.readAsDataURL(blob)
        })

        try {
          const response = await fetch('http://localhost:5000/predict', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image: base64 })
          })
          
          const data = await response.json()
          setDetected(data.sign || 'Waiting for hands...')
          setConfidence(Math.round((data.confidence || 0) * 100))
          
          if (data.keypoints && data.keypoints.length > 0) {
            // Scale keypoints to display resolution
            const scaleX = canvasElement.width / 160
            const scaleY = canvasElement.height / 120
            const scaledKeypoints = data.keypoints.map(hand => 
              hand.map(point => [point[0] * scaleX, point[1] * scaleY])
            )
            setLastKeypoints(scaledKeypoints)
          } else {
            setLastKeypoints(null)
          }
          
          // Update live caption from backend
          if (data.caption !== undefined) {
            setCaption(data.caption)
          }
          
          // Add completed words to transcript
          if (data.sign && !data.sign.includes('Waiting') && !data.sign.includes('detected') && data.confidence > 0.6) {
            // Only add whole words, not individual letters
            if (data.sign.length > 1 && !data.sign.includes('SPACE')) {
              setTranscript(prev => prev.length > 0 ? prev + ' ' + data.sign : data.sign)
            }
          }
        } catch (error) {
          console.error('Prediction error:', error)
        }

        requestInFlight.current = false
        // Immediately queue next frame
        setTimeout(recordingLoop, 16)
      }, 'image/jpeg', 0.5)  // JPEG at 50% quality for faster transfer
    } else {
      if (isActive) setTimeout(recordingLoop, 16)
    }
  }

  const drawHandKeypoints = (ctx, keypoints) => {
    const connections = [
      [0, 1], [1, 2], [2, 3], [3, 4],
      [0, 5], [5, 6], [6, 7], [7, 8],
      [0, 9], [9, 10], [10, 11], [11, 12],
      [0, 13], [13, 14], [14, 15], [15, 16],
      [0, 17], [17, 18], [18, 19], [19, 20],
      [5, 9], [9, 13], [13, 17]
    ]

    keypoints.forEach((hand) => {
      // Draw gradient connections
      ctx.lineWidth = 4
      connections.forEach(([start, end]) => {
        if (start < hand.length && end < hand.length) {
          const gradient = ctx.createLinearGradient(
            hand[start][0], hand[start][1],
            hand[end][0], hand[end][1]
          )
          gradient.addColorStop(0, '#8B5CF6')
          gradient.addColorStop(1, '#06B6D4')
          ctx.strokeStyle = gradient
          ctx.beginPath()
          ctx.moveTo(hand[start][0], hand[start][1])
          ctx.lineTo(hand[end][0], hand[end][1])
          ctx.stroke()
        }
      })

      // Draw glowing keypoints
      hand.forEach((point, idx) => {
        const gradient = ctx.createRadialGradient(
          point[0], point[1], 0,
          point[0], point[1], 12
        )
        gradient.addColorStop(0, idx < 5 ? '#F472B6' : '#22D3EE')
        gradient.addColorStop(0.5, idx < 5 ? 'rgba(244, 114, 182, 0.5)' : 'rgba(34, 211, 238, 0.5)')
        gradient.addColorStop(1, 'transparent')
        ctx.fillStyle = gradient
        ctx.beginPath()
        ctx.arc(point[0], point[1], 12, 0, 2 * Math.PI)
        ctx.fill()
        
        // Inner dot
        ctx.fillStyle = '#fff'
        ctx.beginPath()
        ctx.arc(point[0], point[1], 4, 0, 2 * Math.PI)
        ctx.fill()
      })
    })
  }

  const copyMeetingCode = () => {
    navigator.clipboard.writeText(meetingCode)
    alert('Meeting code copied!')
  }

  const handleJoinMeeting = () => {
    if (!joinCode.trim()) {
      alert('Please enter a meeting code')
      return
    }
    setMeetingCode(joinCode)
    setPendingParticipants([{
      id: user.id,
      name: username || user.username || 'Anonymous',
      joinedAt: new Date()
    }])
    startCall()
  }

  const admitParticipant = (participantId) => {
    const pending = pendingParticipants.find(p => p.id === participantId)
    if (pending) {
      setParticipants([...participants, pending])
      setPendingParticipants(pendingParticipants.filter(p => p.id !== participantId))
    }
  }

  const rejectParticipant = (participantId) => {
    setPendingParticipants(pendingParticipants.filter(p => p.id !== participantId))
  }

  const clearTranscript = () => setTranscript('')

  const downloadTranscript = () => {
    const element = document.createElement('a')
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(transcript))
    element.setAttribute('download', 'transcript.txt')
    element.style.display = 'none'
    document.body.appendChild(element)
    element.click()
    document.body.removeChild(element)
  }

  // Loading screen
  if (!isLoaded) {
    return (
      <div className="app">
        <div className="loading-screen">
          <div className="loading-spinner"></div>
          <div className="loading-text">Loading...</div>
        </div>
      </div>
    )
  }

  // Login screen
  if (!isSignedIn) {
    return (
      <div className="app">
        <div className="login-screen">
          <div className="login-container">
            <div className="login-icon">🤟</div>
            <h1 className="login-title">SignSpeak</h1>
            <p className="login-subtitle">Real-time sign language recognition powered by AI</p>
            <SignIn 
              appearance={{
                elements: {
                  rootBox: { width: '100%', maxWidth: '420px' },
                  card: { background: 'rgba(15, 23, 42, 0.8)', border: '1px solid rgba(255,255,255,0.1)', borderRadius: '24px' },
                  formButtonPrimary: { background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)' },
                  formFieldInput: { background: 'rgba(0,0,0,0.3)', borderColor: 'rgba(255,255,255,0.1)' }
                }
              }}
            />
          </div>
        </div>
      </div>
    )
  }

  // Username setup
  if (isSettingUsername) {
    return (
      <div className="app">
        <div className="page">
          <div className="page-card">
            <div className="page-header">
              <div className="page-icon">👋</div>
              <h1 className="page-title">Welcome!</h1>
              <p className="page-subtitle">Choose a username to get started</p>
            </div>
            
            <div className="form-group">
              <label className="form-label">Username</label>
              <input
                type="text"
                className="form-input"
                placeholder="Enter your username"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
            </div>
            
            <button 
              className="btn btn-gradient" 
              style={{ width: '100%' }}
              onClick={() => {
                if (username.trim()) {
                  setIsSettingUsername(false)
                  setScreen('home')
                } else {
                  alert('Username cannot be empty')
                }
              }}
            >
              <span className="icon">🚀</span>
              Continue
            </button>
          </div>
        </div>
      </div>
    )
  }

  return (
    <div className="app">
      {/* Hidden video element */}
      <video 
        ref={videoRef}
        autoPlay 
        playsInline
        muted
        style={{ 
          position: 'absolute',
          opacity: screen === 'call' ? 1 : 0,
          pointerEvents: 'none',
          width: '1px',
          height: '1px'
        }}
      />
      
      {/* Navbar - show on all pages except call */}
      {screen !== 'call' && (
        <nav className="navbar">
          <div className="nav-brand">
            <span className="logo-icon">🤟</span>
            <span>SignSpeak</span>
          </div>
          
          <div className="nav-links">
            <button 
              className={`nav-link ${screen === 'home' ? 'active' : ''}`}
              onClick={() => setScreen('home')}
            >
              Home
            </button>
            <button 
              className={`nav-link ${screen === 'create' ? 'active' : ''}`}
              onClick={() => setScreen('create')}
            >
              Create Meeting
            </button>
            <button 
              className={`nav-link ${screen === 'join' ? 'active' : ''}`}
              onClick={() => setScreen('join')}
            >
              Join Meeting
            </button>
            <button 
              className={`nav-link ${screen === 'about' ? 'active' : ''}`}
              onClick={() => setScreen('about')}
            >
              About
            </button>
          </div>
          
          <div className="nav-user">
            <span className="nav-username">{username || user?.username}</span>
            <UserButton />
          </div>
        </nav>
      )}

      {/* HOME PAGE */}
      {screen === 'home' && (
        <>
          <div className="hero-page">
            <div className="hero-content">
              <div className="hero-badge">
                <span className="dot"></span>
                <span>AI-Powered Recognition</span>
              </div>
              
              <div className="hero-icon">🤟</div>
              
              <h1 className="hero-title">
                Breaking Barriers<br />Through Signs
              </h1>
              
              <p className="hero-subtitle">
                Experience seamless communication with our real-time sign language 
                recognition technology. Connect, communicate, and collaborate without limits.
              </p>
              
              <div className="hero-buttons">
                <button className="btn btn-gradient" onClick={() => setScreen('create')}>
                  <span className="icon">🎬</span>
                  Start Meeting
                </button>
                <button className="btn btn-secondary" onClick={() => setScreen('join')}>
                  <span className="icon">🔗</span>
                  Join Meeting
                </button>
              </div>
              
              <div className="status-badge">
                <span className={`status-dot ${backendStatus === 'Connected' ? 'connected' : 'disconnected'}`}></span>
                <span>AI Engine:</span>
                <span className={`status-text ${backendStatus === 'Connected' ? 'connected' : 'disconnected'}`}>
                  {backendStatus}
                </span>
              </div>
            </div>
          </div>
          
          <div className="features-section">
            <h2 className="section-title">Why SignSpeak?</h2>
            <p className="section-subtitle">Powerful features designed for seamless communication</p>
            
            <div className="features-grid">
              <div className="feature-card" onClick={() => setScreen('create')}>
                <div className="feature-icon">🎥</div>
                <h3 className="feature-title">Video Meetings</h3>
                <p className="feature-desc">
                  High-quality video conferencing with real-time sign language detection and transcription.
                </p>
              </div>
              
              <div className="feature-card" onClick={() => setScreen('create')}>
                <div className="feature-icon">🤖</div>
                <h3 className="feature-title">AI Recognition</h3>
                <p className="feature-desc">
                  Advanced machine learning models detect hand gestures with incredible accuracy.
                </p>
              </div>
              
              <div className="feature-card" onClick={() => setScreen('create')}>
                <div className="feature-icon">📝</div>
                <h3 className="feature-title">Live Transcription</h3>
                <p className="feature-desc">
                  Automatic transcription of detected signs into readable text in real-time.
                </p>
              </div>
              
              <div className="feature-card" onClick={() => setScreen('profile')}>
                <div className="feature-icon">👥</div>
                <h3 className="feature-title">Team Collaboration</h3>
                <p className="feature-desc">
                  Invite participants, manage meetings, and collaborate seamlessly.
                </p>
              </div>
            </div>
          </div>
        </>
      )}

      {/* CREATE MEETING PAGE */}
      {screen === 'create' && (
        <div className="page">
          <div className="page-card">
            <div className="page-header">
              <div className="page-icon">🎬</div>
              <h1 className="page-title">Create Meeting</h1>
              <p className="page-subtitle">Share the code with participants</p>
            </div>
            
            <div className="form-group">
              <label className="form-label">Meeting Code</label>
              <div className="input-group">
                <input 
                  type="text" 
                  className="form-input code" 
                  value={meetingCode} 
                  readOnly 
                />
                <button className="copy-btn" onClick={copyMeetingCode}>📋 Copy</button>
              </div>
            </div>
            
            <div className="checkbox-group">
              <input 
                type="checkbox" 
                id="muted"
                checked={isMuted}
                onChange={(e) => setIsMuted(e.target.checked)}
              />
              <label htmlFor="muted">Start with microphone muted</label>
            </div>
            
            <button className="btn btn-gradient" style={{ width: '100%' }} onClick={startCall}>
              <span className="icon">📞</span>
              Start Meeting
            </button>
            
            <button className="back-btn" onClick={() => setScreen('home')}>
              ← Back to Home
            </button>
            
            <div className="status-badge">
              <span className={`status-dot ${backendStatus === 'Connected' ? 'connected' : 'disconnected'}`}></span>
              <span>AI:</span>
              <span className={`status-text ${backendStatus === 'Connected' ? 'connected' : 'disconnected'}`}>
                {backendStatus}
              </span>
            </div>
          </div>
        </div>
      )}

      {/* JOIN MEETING PAGE */}
      {screen === 'join' && (
        <div className="page">
          <div className="page-card">
            <div className="page-header">
              <div className="page-icon">🔗</div>
              <h1 className="page-title">Join Meeting</h1>
              <p className="page-subtitle">Enter the meeting code to join</p>
            </div>
            
            <div className="form-group">
              <label className="form-label">Meeting Code</label>
              <input 
                type="text" 
                className="form-input code" 
                placeholder="e.g., SL-ABC123XYZ"
                value={joinCode}
                onChange={(e) => setJoinCode(e.target.value.toUpperCase())}
                onKeyPress={(e) => e.key === 'Enter' && handleJoinMeeting()}
              />
            </div>
            
            <div className="checkbox-group">
              <input 
                type="checkbox" 
                id="muted-join"
                checked={isMuted}
                onChange={(e) => setIsMuted(e.target.checked)}
              />
              <label htmlFor="muted-join">Join with microphone muted</label>
            </div>
            
            <button className="btn btn-primary" style={{ width: '100%' }} onClick={handleJoinMeeting}>
              <span className="icon">🚀</span>
              Join Meeting
            </button>
            
            <button className="back-btn" onClick={() => setScreen('home')}>
              ← Back to Home
            </button>
          </div>
        </div>
      )}

      {/* ABOUT PAGE */}
      {screen === 'about' && (
        <div className="page">
          <div className="page-card" style={{ maxWidth: '600px' }}>
            <div className="page-header">
              <div className="page-icon">ℹ️</div>
              <h1 className="page-title">About SignSpeak</h1>
              <p className="page-subtitle">Empowering communication through technology</p>
            </div>
            
            <div style={{ color: 'var(--text-muted)', lineHeight: '1.8', marginBottom: '24px' }}>
              <p style={{ marginBottom: '16px' }}>
                <strong style={{ color: 'var(--secondary-light)' }}>SignSpeak</strong> is an innovative 
                platform that uses artificial intelligence to recognize sign language in real-time during 
                video calls.
              </p>
              <p style={{ marginBottom: '16px' }}>
                Our mission is to break down communication barriers and make video conferencing accessible 
                to everyone, regardless of how they communicate.
              </p>
              <p>
                Built with ❤️ using React, Python, MediaPipe, and machine learning.
              </p>
            </div>
            
            <div style={{ 
              display: 'grid', 
              gridTemplateColumns: 'repeat(3, 1fr)', 
              gap: '16px', 
              marginBottom: '24px' 
            }}>
              <div style={{ 
                textAlign: 'center', 
                padding: '20px', 
                background: 'var(--glass)', 
                borderRadius: '16px' 
              }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>🤖</div>
                <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--primary-light)' }}>AI</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Powered</div>
              </div>
              <div style={{ 
                textAlign: 'center', 
                padding: '20px', 
                background: 'var(--glass)', 
                borderRadius: '16px' 
              }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>⚡</div>
                <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--secondary-light)' }}>Real</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Time</div>
              </div>
              <div style={{ 
                textAlign: 'center', 
                padding: '20px', 
                background: 'var(--glass)', 
                borderRadius: '16px' 
              }}>
                <div style={{ fontSize: '32px', marginBottom: '8px' }}>🌍</div>
                <div style={{ fontSize: '24px', fontWeight: '800', color: 'var(--accent-light)' }}>Open</div>
                <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>Source</div>
              </div>
            </div>
            
            <button className="back-btn" onClick={() => setScreen('home')}>
              ← Back to Home
            </button>
          </div>
        </div>
      )}

      {/* PROFILE PAGE */}
      {screen === 'profile' && (
        <div className="page">
          <div className="page-card">
            <div className="page-header">
              <div className="page-icon">👤</div>
              <h1 className="page-title">Your Profile</h1>
            </div>
            
            <div className="profile-avatar-section">
              <img src={user?.imageUrl} alt="Avatar" className="profile-avatar" />
              <p className="profile-email">{user?.emailAddresses[0]?.emailAddress}</p>
            </div>
            
            <div className="form-group">
              <label className="form-label">Username</label>
              <input
                type="text"
                className="form-input"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Enter username"
              />
            </div>
            
            <button 
              className="btn btn-gradient" 
              style={{ width: '100%' }} 
              onClick={() => setScreen('home')}
            >
              <span className="icon">💾</span>
              Save & Go Home
            </button>
            
            <button className="back-btn" onClick={() => setScreen('home')}>
              ← Cancel
            </button>
          </div>
        </div>
      )}

      {/* CALL VIEW */}
      {screen === 'call' && (
        <div className="call-view">
          {/* Header */}
          <div className="call-header">
            <div className="call-info">
              <span className="call-code">{meetingCode}</span>
              <span className="call-participants-count">
                <span>{participants.length}</span> participant{participants.length !== 1 ? 's' : ''}
              </span>
            </div>
            <button className="btn btn-danger" onClick={endCall}>
              <span className="icon">📞</span>
              End Call
            </button>
          </div>

          {/* Main video area */}
          <div className="main-video-area">
            <div className="video-wrapper">
              {isVideoOff ? (
                <div className="video-off-placeholder">
                  <span className="icon">📷</span>
                  <span className="text">Camera is off</span>
                </div>
              ) : (
                <canvas 
                  ref={canvasRef}
                  width="1280"
                  height="720"
                  className="main-canvas"
                />
              )}
              
              {/* Detection overlay */}
              <div className="detection-overlay">
                <div className="detection-card">
                  <div className="detected-sign">{detected}</div>
                  <div className="confidence-bar">
                    <div className="confidence-fill" style={{ width: confidence + '%' }}></div>
                  </div>
                  <span className="confidence-text">{confidence}% confidence</span>
                </div>
              </div>

              {/* Live Caption Display */}
              {caption && (
                <div className="caption-overlay">
                  <div className="caption-text">{caption}</div>
                  <div className="caption-controls">
                    <button onClick={backspace} title="Backspace">⌫</button>
                    <button onClick={addSpace} title="Add Space">␣</button>
                    <button onClick={clearCaption} title="Clear">✕</button>
                  </div>
                </div>
              )}

              {/* Gesture popup */}
              {detected && (detected.includes('LOVE') || detected.includes('HELLO')) && (
                <div className="gesture-popup">
                  {detected === 'LOVE' ? '💜 LOVE' : '👋 HELLO'}
                </div>
              )}

              {/* Video controls */}
              <div className="video-controls">
                <button 
                  className={`control-btn ${!isMuted ? 'active' : 'inactive'}`}
                  onClick={() => setIsMuted(!isMuted)}
                  title={isMuted ? 'Unmute' : 'Mute'}
                >
                  {isMuted ? '🔇' : '🔊'}
                </button>
                <button 
                  className={`control-btn ${!isVideoOff ? 'active' : 'inactive'}`}
                  onClick={() => setIsVideoOff(!isVideoOff)}
                  title={isVideoOff ? 'Start video' : 'Stop video'}
                >
                  {isVideoOff ? '📷' : '🎥'}
                </button>
              </div>

              {/* Participant label */}
              <div className="participant-label">
                {username || user?.username || 'You'}
                {isHost && <span className="host-badge">(Host)</span>}
              </div>
            </div>

            {/* Participant strip */}
            {participants.length > 1 && (
              <div className="participants-strip">
                {participants.slice(1).map((p) => (
                  <div key={p.id} className="participant-tile">
                    <span className="icon">👤</span>
                    <span className="name">{p.name}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Sidebar */}
          <div className="call-sidebar">
            {/* Pending participants */}
            {isHost && pendingParticipants.length > 0 && (
              <div className="sidebar-panel">
                <div className="panel-header">
                  <span className="icon">🔔</span>
                  <span>Waiting to Join</span>
                </div>
                <div className="pending-list">
                  {pendingParticipants.map((p) => (
                    <div key={p.id} className="pending-item">
                      <span className="name">{p.name}</span>
                      <div className="pending-actions">
                        <button className="admit-btn" onClick={() => admitParticipant(p.id)}>✓</button>
                        <button className="reject-btn" onClick={() => rejectParticipant(p.id)}>✗</button>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {/* Participants */}
            <div className="sidebar-panel">
              <div className="panel-header">
                <span className="icon">👥</span>
                <span>Participants ({participants.length})</span>
              </div>
              <div className="participants-list">
                {participants.map((p) => (
                  <div key={p.id} className="participant-row">
                    <span className="name">{p.name}</span>
                    {p.isHost && <span className="badge">Host</span>}
                  </div>
                ))}
              </div>
            </div>

            {/* Transcript */}
            <div className="sidebar-panel">
              <div className="panel-header">
                <span className="icon">📝</span>
                <span>Live Transcript</span>
              </div>
              <div className="transcript-box">
                {transcript ? (
                  <span className="transcript-text">{transcript}</span>
                ) : (
                  <span className="transcript-placeholder">No signs detected yet...</span>
                )}
              </div>
              <div className="transcript-actions">
                <button className="transcript-btn" onClick={clearTranscript}>Clear</button>
                <button className="transcript-btn" onClick={downloadTranscript}>Download</button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
