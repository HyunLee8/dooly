'use client'
import { useState } from 'react';
import Orb from '@/components/Orb';
import TextType from './TextType';

export default function AudioReq() {
  const [isActive, setIsActive] = useState(false);
  const [showPrompt, setShowPrompt] = useState(false);

  const handleActivate = async () => {
    setIsActive(true);
    setTimeout(() => {
      setShowPrompt(true);
    }, 1000);
    try {
      await navigator.mediaDevices.getUserMedia({ audio: true });
    } catch (err) {
      console.error("Microphone access denied:", err);
    }
  }

  return (
    <div className="gap-5 flex flex-col items-center">
      <div className="mt-10" style={{ position: 'absolute', inset: 0, zIndex: 0 }}>
        <Orb
          hoverIntensity={1.0}
          rotateOnHover={true}
          hue={0}
          forceHoverState={false}
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
              text={["How may I assist you today?"]}
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