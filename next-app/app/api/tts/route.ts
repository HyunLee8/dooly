import { NextResponse } from 'next/server';

export async function POST(req: Request) {
  const { text } = await req.json();
  const VOICE_ID = 'TVtDNgumMv4lb9zzFzA2';
  
  const response = await fetch(
    `https://api.elevenlabs.io/v1/text-to-speech/${VOICE_ID}`,
    {
      method: 'POST',
      headers: {
        'xi-api-key': process.env.ELEVENLABS_API_KEY || '',
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        text: text,
        model_id: 'eleven_monolingual_v1',
        output_format: 'mp3_44100_128'
      })
    }
  );
  
  const audioBlob = await response.blob();
  return new NextResponse(audioBlob, {
    headers: { 'Content-Type': 'audio/mpeg' }
  });
}