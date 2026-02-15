"use client";
import { useState } from "react";
import ColorScale from "~/components/ColorScale";
import FeatureMap from "~/components/FeatureMap";
import { Badge } from "~/components/ui/badge";
import { Button } from "~/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "~/components/ui/card";
import { Progress } from "~/components/ui/progress";
import Waveform from "../components/Waveform";

interface Prediction {
  class: string;
  confidence: number;
}

interface LayerData {
  shape: number[];
  values: number[][];
}

interface WaveformData {
  values: number[];
  sample_rate: number;
  duration: number;
}

interface ApiResponse {
  predictions: Prediction[];
  input_spectrogram: LayerData;
  waveform: WaveformData;
}

const ESC50_EMOJI_MAP: Record<string, string> = {
  dog: "🐕",
  rain: "🌧️",
  crying_baby: "👶",
  helicopter: "🚁",
  rooster: "🐓",
  sea_waves: "🌊",
  mouse_click: "🖱️",
  chainsaw: "🪚",
  pig: "🐷",
  crackling_fire: "🔥",
  clapping: "👏",
  keyboard_typing: "⌨️",
  siren: "🚨",
  cow: "🐄",
  crickets: "🦗",
  breathing: "💨",
  car_horn: "📯",
  frog: "🐸",
  chirping_birds: "🐦",
  coughing: "😷",
  engine: "🚗",
  cat: "🐱",
  water_drops: "💧",
  footsteps: "👣",
  train: "🚂",
  wind: "💨",
  laughing: "😂",
  church_bells: "🔔",
  pouring_water: "🚰",
  brushing_teeth: "🪥",
  clock_alarm: "⏰",
  airplane: "✈️",
  sheep: "🐑",
  toilet_flush: "🚽",
  snoring: "😴",
  clock_tick: "⏱️",
  fireworks: "🎆",
  thunderstorm: "⛈️",
  drinking_sipping: "🥤",
  glass_breaking: "🔨",
};

const getEmojiForClass = (className: string): string =>
  ESC50_EMOJI_MAP[className] || "🔈";

export default function HomePage() {
  const [vizData, setVizData] = useState<ApiResponse | null>(null);
  const [hasUploaded, setHasUploaded] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [error, setError] = useState<string | null>(null);

  const handleFileChange = async (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setVizData(null);
    setHasUploaded(false);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);

    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;

        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            "",
          ),
        );

        const response = await fetch(
          "https://imrealpawan--audio-cnn-inference-audioclassifier-inference.modal.run/",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ audio_data: base64String }),
          },
        );

        if (!response.ok) {
          throw new Error(`API error ${response.statusText}`);
        }

        const data: ApiResponse = await response.json();
        console.log("FULL API RESPONSE:", data);

        setVizData(data);
        setHasUploaded(true);
      } catch (err) {
        setError(
          err instanceof Error ? err.message : "An unknown error occurred",
        );
      } finally {
        setIsLoading(false);
      }
    };

    reader.onerror = () => {
      setError("Failed to read the file.");
      setIsLoading(false);
    };
  };

  return (
    <main className="min-h-screen bg-stone-50 p-8">
      <div className="mx-auto max-w-[60%]">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-4xl font-light tracking-tight text-stone-900">
            CNN Audio Visualizer
          </h1>
          <p className="mb-8 text-lg text-stone-600">
            Upload a WAV/MP3 file to see predictions and audio visuals
          </p>

          <div className="flex flex-col items-center">
            <div className="relative inline-block">
              <input
                type="file"
                accept=".wav,.mp3"
                disabled={isLoading}
                onChange={handleFileChange}
                className="absolute inset-0 w-full cursor-pointer opacity-0"
              />
              <Button
                disabled={isLoading}
                variant="outline"
                size="lg"
                className="border-stone-300"
              >
                {isLoading ? "Analysing..." : "Choose File"}
              </Button>
            </div>

            {fileName && (
              <Badge className="mt-4 bg-stone-200 text-stone-700">
                {fileName}
              </Badge>
            )}
          </div>
        </div>

        {error && (
          <Card className="mb-8 border-red-200 bg-red-50">
            <CardContent>
              <p className="text-red-600">Error: {error}</p>
            </CardContent>
          </Card>
        )}

        {hasUploaded && vizData && (
          <div className="space-y-8">
            {/* Predictions */}
            <Card>
              <CardHeader>
                <CardTitle>Top Predictions</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-3">
                  {vizData.predictions.map((pred, i) => (
                    <div key={pred.class} className="space-y-2">
                      <div className="flex justify-between">
                        <div className="font-medium text-stone-700">
                          {getEmojiForClass(pred.class)}{" "}
                          {pred.class.replaceAll("_", " ")}
                        </div>
                        <Badge variant={i === 0 ? "default" : "secondary"}>
                          {(pred.confidence * 100).toFixed(1)}%
                        </Badge>
                      </div>
                      <Progress value={pred.confidence * 100} className="h-2" />
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            {/* Spectrogram + Waveform */}
            <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
              <Card>
                <CardHeader>
                  <CardTitle>Input Spectrogram</CardTitle>
                </CardHeader>
                <CardContent>
                  {vizData.input_spectrogram?.values?.length > 0 && (
                    <FeatureMap
                      data={vizData.input_spectrogram.values}
                      title={`${vizData.input_spectrogram.shape.join(" x ")}`}
                      spectrogram
                    />
                  )}
                  <div className="mt-5 flex justify-end">
                    <ColorScale width={200} height={16} min={-1} max={1} />
                  </div>
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle>Audio Waveform</CardTitle>
                </CardHeader>
                <CardContent>
                  {vizData.waveform?.values?.length > 0 && (
                    <Waveform
                      data={vizData.waveform.values}
                      title={`${vizData.waveform.duration.toFixed(2)}s • ${
                        vizData.waveform.sample_rate
                      }Hz`}
                    />
                  )}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </main>
  );
}
