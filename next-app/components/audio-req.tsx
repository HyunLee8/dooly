/*
I sowwy for anyone looking at this code because
it's absolutely horrendous. I litteraly put everything in one
component lol. I'll refactor this later I promise.
*/

'use client'

import { useState } from 'react';
import Orb from '@/components/Orb';
import TextType from './TextType';


//fix this ugly ass code later
async function SpeakWithElevenLabs(text: string) {
    const response = await fetch('/api/speech', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text }),
    });

    const audioBlob = await response.blob();
    const audioUrl = URL.createObjectURL(audioBlob);
    
    const audio = new Audio(audioUrl);
    await audio.play();
    
    // Clean up
    audio.onended = () => URL.revokeObjectURL(audioUrl);
}

export default function AudioReq() {
  const [isActive, setIsActive] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);
  const [mediaRecorder, setMediaRecorder] = useState<MediaRecorder | null>(null);
  const [audioDetected, setAudioDetected] = useState(false);
  const [currentMessage, setCurrentMessage] = useState('');
  const [thinking, setThinking] = useState(false);

  const handleActivate = async () => {
    setIsActive(true);
    setTimeout(() => setShowPrompt(true), 1000);

    try {
      SpeakWithElevenLabs('How may I assist you today?');
      await new Promise(resolve => setTimeout(resolve, 1000));
      setCurrentMessage('How may I assist you today?');
      await new Promise(resolve => setTimeout(resolve, 4000));

      const stream = await navigator.mediaDevices.getUserMedia({ audio: true });

      // Setup silence detection
      const audioContext = new AudioContext();
      const analyser = audioContext.createAnalyser();
      const microphone = audioContext.createMediaStreamSource(stream);
      analyser.fftSize = 256;
      microphone.connect(analyser);

      const dataArray = new Uint8Array(analyser.frequencyBinCount);
      let silenceStart = Date.now();
      const SILENCE_THRESHOLD = 0.15;
      const SILENCE_DURATION = 2000;

      // starts recording
      const recorder = new MediaRecorder(stream);
      const chunks: Blob[] = [];
      recorder.ondataavailable = (e) => chunks.push(e.data);

      recorder.onstop = async () => {
        const audioBlob = new Blob(chunks, { type: 'audio/webm' });
        setCurrentMessage('Processing audio...');
        const formData = new FormData();
        formData.append('audio', audioBlob, 'recording.webm');

        const response = await fetch('http://localhost:5000/api/process', {
          method: 'POST',
          body: formData,
        });

        const data = await response.json();
        setThinking(true);
        setCurrentMessage('reporting back...');
        SpeakWithElevenLabs(data.message);
        audioContext.close();
      };

      recorder.start();
      setMediaRecorder(recorder);

      const checkSilence = () => {
        analyser.getByteFrequencyData(dataArray);
        const average = dataArray.reduce((a, b) => a + b) / dataArray.length / 255;

        if (average < SILENCE_THRESHOLD) {
          setAudioDetected(false);
          if (Date.now() - silenceStart > SILENCE_DURATION) {
            recorder.stop();
            return;
          }
        } else {
          setAudioDetected(true);
          silenceStart = Date.now();
        }

        requestAnimationFrame(checkSilence);
      };

      checkSilence();

    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  };

  return (
    <div className="gap-5 flex flex-col items-center">
      {showPrompt && isActive && (
        <p className="fixed text-xl top-4 right-4">
          {audioDetected ? (
            <span className="text-green-500">● audio detected</span>
          ) : (
            <span className="text-red-500">● no audio detected</span>
          )}
        </p>
      )}
      <div className="mt-10" style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <Orb
          key={thinking ? 'thinking' : 'idle'}
          hoverIntensity={1.0}
          rotateOnHover={true}
          hue={0}
          forceHoverState={thinking}
        />
      </div>
      <div className="z-10 text-center relative">
        <div className={`transition-opacity duration-1000 ${isActive ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}>
          <div className="flex flex-col gap-5">
            <h1 className="text-4xl">Hello</h1>
            <h2 className="text-7xl">{"I'm DOOLY"}</h2>
            <button onClick={handleActivate} className="p-5 border bg-white hover:bg-black hover:text-white duration-500 z-10">Click here to activate me</button>
          </div>
        </div>
        {showPrompt && (
          <h2 className="min-w-max text-3xl absolute top-1/2 left-1/2 -translate-x-1/2 animate-fade-in">
            <TextType
              key={currentMessage}
              text={[currentMessage]}
              typingSpeed={75}
              pauseDuration={1500}
              showCursor={true}
              cursorCharacter="_"
            />
          </h2>
        )}
      </div>
    </div>
  )
}
