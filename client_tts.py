#!/usr/bin/env python3
"""
TARS TTS Server Client

A comprehensive client for the TARS TTS server that demonstrates all endpoints
with elegant logging and real-time audio playback for streaming responses.

Usage:
    uv run python client_tts.py                    # Run all tests
    uv run python client_tts.py -i                 # Interactive mode
    uv run python client_tts.py --text "Hello"     # Custom text
    uv run python client_tts.py --no-playback      # Skip audio
"""

import argparse
import io
import os
import sys
import time
import wave
from datetime import datetime
from typing import Optional

import numpy as np
import requests

# =============================================================================
# ANSI Color Codes & Box Drawing
# =============================================================================

COLORS = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "cyan": "\033[36m",
    "red": "\033[31m",
    "magenta": "\033[35m",
    "white": "\033[37m",
}

BOX = {
    "top_left": "┌",
    "top_right": "┐",
    "bottom_left": "└",
    "bottom_right": "┘",
    "horizontal": "─",
    "vertical": "│",
    "t_right": "├",
    "t_left": "┤",
}

STATUS = {
    "progress": f"{COLORS['yellow']}[...]{COLORS['reset']}",
    "complete": f"{COLORS['green']}[OK]{COLORS['reset']}",
    "failed": f"{COLORS['red']}[FAIL]{COLORS['reset']}",
    "info": f"{COLORS['blue']}[i]{COLORS['reset']}",
    "request": f"{COLORS['magenta']}[REQ]{COLORS['reset']}",
    "response": f"{COLORS['cyan']}[RES]{COLORS['reset']}",
    "play": f"{COLORS['green']}[▶]{COLORS['reset']}",
    "skip": f"{COLORS['dim']}[SKIP]{COLORS['reset']}",
}

# =============================================================================
# Logging Utilities
# =============================================================================

def print_header(title: str, width: int = 60) -> None:
    """Print a section header with box-drawing characters."""
    title_with_padding = f" {title} "
    remaining = width - len(title_with_padding) - 2
    left_pad = remaining // 2
    right_pad = remaining - left_pad

    line = (
        f"{COLORS['cyan']}{BOX['top_left']}"
        f"{BOX['horizontal'] * left_pad}"
        f"{COLORS['bold']}{title_with_padding}{COLORS['reset']}{COLORS['cyan']}"
        f"{BOX['horizontal'] * right_pad}"
        f"{BOX['top_right']}{COLORS['reset']}"
    )
    print(line)


def print_footer(width: int = 60) -> None:
    """Print a section footer with box-drawing characters."""
    line = (
        f"{COLORS['cyan']}{BOX['bottom_left']}"
        f"{BOX['horizontal'] * (width - 2)}"
        f"{BOX['bottom_right']}{COLORS['reset']}"
    )
    print(line)


def print_item(key: str, value, indent: int = 2) -> None:
    """Print a key-value pair within a section."""
    spaces = " " * indent
    print(
        f"{COLORS['cyan']}{BOX['vertical']}{COLORS['reset']}{spaces}"
        f"{COLORS['dim']}{key}:{COLORS['reset']} {COLORS['white']}{value}{COLORS['reset']}"
    )


def print_stage(message: str, status: str = "progress", elapsed: Optional[float] = None) -> None:
    """Print a formatted stage message with status indicator."""
    symbol = STATUS.get(status, STATUS["info"])
    parts = [symbol, message]

    if elapsed is not None:
        parts.append(f"{COLORS['dim']}({elapsed:.2f}s){COLORS['reset']}")

    print(" ".join(parts))


def format_bytes(num_bytes: int) -> str:
    """Format bytes to human-readable string."""
    if num_bytes < 1024:
        return f"{num_bytes} B"
    elif num_bytes < 1024 * 1024:
        return f"{num_bytes / 1024:.1f} KB"
    else:
        return f"{num_bytes / (1024 * 1024):.1f} MB"


def format_duration(seconds: float) -> str:
    """Format duration to readable string."""
    if seconds < 1:
        return f"{seconds * 1000:.0f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        mins = int(seconds // 60)
        secs = seconds % 60
        return f"{mins}m {secs:.1f}s"


# =============================================================================
# Audio Playback
# =============================================================================

def get_audio_player():
    """Get a cross-platform audio player. Returns (player, cleanup_func) or (None, None)."""
    try:
        import sounddevice as sd
        return sd, None
    except ImportError:
        return None, None
    except OSError:
        # PortAudio library not found
        return None, None


def play_audio_bytes(audio_bytes: bytes, sample_rate: int = 22050, is_wav: bool = True) -> None:
    """Play audio from bytes using sounddevice."""
    sd, _ = get_audio_player()
    if sd is None:
        print_stage("Audio playback unavailable (install sounddevice)", "skip")
        return

    try:
        if is_wav:
            # Parse WAV header to get audio data
            with io.BytesIO(audio_bytes) as wav_io:
                with wave.open(wav_io, 'rb') as wav_file:
                    sample_rate = wav_file.getframerate()
                    audio_data = np.frombuffer(wav_file.readframes(wav_file.getnframes()), dtype=np.int16)
        else:
            # Raw PCM S16LE
            audio_data = np.frombuffer(audio_bytes, dtype=np.int16)

        # Normalize to float32 for sounddevice
        audio_float = audio_data.astype(np.float32) / 32768.0

        print_stage("Playing audio...", "play")
        sd.play(audio_float, sample_rate)
        sd.wait()
        print_stage("Playback complete", "complete")
    except Exception as e:
        print_stage(f"Playback failed: {e}", "failed")


class StreamingAudioPlayer:
    """Plays audio chunks in real-time as they arrive."""

    def __init__(self, sample_rate: int = 22050, channels: int = 1):
        self.sample_rate = sample_rate
        self.channels = channels
        self.sd, _ = get_audio_player()
        self.stream = None
        self.buffer = []
        self.started = False

    def available(self) -> bool:
        """Check if audio playback is available."""
        return self.sd is not None

    def start(self) -> bool:
        """Start the audio stream."""
        if not self.available():
            print_stage("Streaming playback unavailable (install sounddevice)", "skip")
            return False

        try:
            self.stream = self.sd.OutputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype=np.float32,
            )
            self.stream.start()
            self.started = True
            print_stage("Audio stream started", "play")
            return True
        except Exception as e:
            print_stage(f"Failed to start audio stream: {e}", "failed")
            return False

    def write_chunk(self, chunk: bytes) -> None:
        """Write a chunk of PCM S16LE audio data to the stream."""
        if not self.started or self.stream is None:
            return

        try:
            # Convert S16LE bytes to float32
            audio_data = np.frombuffer(chunk, dtype=np.int16)
            audio_float = audio_data.astype(np.float32) / 32768.0

            # Write to stream
            self.stream.write(audio_float)
        except Exception as e:
            print_stage(f"Chunk write failed: {e}", "failed")

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self.stream is not None:
            try:
                self.stream.stop()
                self.stream.close()
                print_stage("Audio stream stopped", "complete")
            except Exception:
                pass
            finally:
                self.stream = None
                self.started = False


# =============================================================================
# API Client
# =============================================================================

class TARSClient:
    """Client for the TARS TTS Server API."""

    def __init__(self, base_url: str = "http://localhost:8009"):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def healthz(self) -> dict:
        """Check server health."""
        resp = self.session.get(f"{self.base_url}/healthz", timeout=5)
        resp.raise_for_status()
        return resp.json()

    def readyz(self) -> dict:
        """Check server readiness."""
        resp = self.session.get(f"{self.base_url}/readyz", timeout=10)
        resp.raise_for_status()
        return resp.json()

    def tts(
        self,
        text: str,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        speed: float = 1.0,
    ) -> bytes:
        """Generate speech from text (blocking, returns WAV bytes)."""
        payload = {
            "text": text,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "speed": speed,
        }
        resp = self.session.post(f"{self.base_url}/tts", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.content

    def tts_stream(
        self,
        text: str,
        temperature: float = 0.8,
        top_p: float = 0.8,
        top_k: int = 30,
        speed: float = 1.0,
    ):
        """Generate speech from text (streaming, yields PCM chunks)."""
        payload = {
            "text": text,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "speed": speed,
        }
        resp = self.session.post(
            f"{self.base_url}/tts/stream",
            json=payload,
            stream=True,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.iter_content(chunk_size=None)

    def wait_for_ready(self, max_retries: int = 10, retry_delay: float = 2.0) -> bool:
        """Wait for the server to be ready."""
        for attempt in range(max_retries):
            try:
                self.healthz()
                return True
            except requests.exceptions.ConnectionError:
                if attempt < max_retries - 1:
                    print_stage(f"Server not ready, retrying ({attempt + 1}/{max_retries})...", "progress")
                    time.sleep(retry_delay)
        return False


# =============================================================================
# Test Functions
# =============================================================================

def test_healthz(client: TARSClient) -> bool:
    """Test the /healthz endpoint."""
    print()
    print_header("Testing /healthz", 50)

    try:
        start = time.perf_counter()
        result = client.healthz()
        elapsed = time.perf_counter() - start

        print_item("Status", result.get("status", "unknown"))
        print_item("Latency", format_duration(elapsed))
        print_footer(50)
        print_stage("/healthz passed", "complete")
        return True
    except Exception as e:
        print_item("Error", str(e))
        print_footer(50)
        print_stage(f"/healthz failed: {e}", "failed")
        return False


def test_readyz(client: TARSClient) -> bool:
    """Test the /readyz endpoint."""
    print()
    print_header("Testing /readyz", 50)

    try:
        start = time.perf_counter()
        result = client.readyz()
        elapsed = time.perf_counter() - start

        ready = result.get("ready", False)
        status_color = COLORS["green"] if ready else COLORS["red"]
        ready_str = f"{status_color}{ready}{COLORS['reset']}"

        print_item("Ready", ready_str)
        print_item("Device", result.get("device", "unknown"))
        print_item("Ref Audio", os.path.basename(result.get("ref_audio", "unknown")))
        print_item("Latency", format_duration(elapsed))
        print_footer(50)
        print_stage("/readyz passed", "complete")
        return True
    except Exception as e:
        print_item("Error", str(e))
        print_footer(50)
        print_stage(f"/readyz failed: {e}", "failed")
        return False


def test_tts(client: TARSClient, text: str, play_audio: bool = True) -> bool:
    """Test the /tts endpoint (blocking)."""
    print()
    print_header("Testing /tts (blocking)", 50)

    truncated = text[:40] + "..." if len(text) > 40 else text
    print_item("Text", f'"{truncated}"')

    try:
        start = time.perf_counter()
        audio_bytes = client.tts(text)
        elapsed = time.perf_counter() - start

        # Parse WAV to get audio info
        with io.BytesIO(audio_bytes) as wav_io:
            with wave.open(wav_io, 'rb') as wav_file:
                sample_rate = wav_file.getframerate()
                n_frames = wav_file.getnframes()
                duration = n_frames / sample_rate

        rtf = elapsed / duration if duration > 0 else 0
        rtf_color = COLORS["green"] if rtf < 1.0 else COLORS["yellow"] if rtf < 2.0 else COLORS["reset"]

        print_item("Size", format_bytes(len(audio_bytes)))
        print_item("Sample Rate", f"{sample_rate} Hz")
        print_item("Duration", format_duration(duration))
        print_item("Latency", format_duration(elapsed))
        print_item("RTF", f"{rtf_color}{rtf:.2f}{COLORS['reset']}")
        print_footer(50)
        print_stage("/tts passed", "complete")

        if play_audio:
            play_audio_bytes(audio_bytes, sample_rate, is_wav=True)

        return True
    except Exception as e:
        print_item("Error", str(e))
        print_footer(50)
        print_stage(f"/tts failed: {e}", "failed")
        return False


def test_tts_stream(client: TARSClient, text: str, play_audio: bool = True) -> bool:
    """Test the /tts/stream endpoint with real-time audio playback."""
    print()
    print_header("Testing /tts/stream (streaming)", 50)

    truncated = text[:40] + "..." if len(text) > 40 else text
    print_item("Text", f'"{truncated}"')
    print_footer(50)

    # Initialize streaming audio player
    player = StreamingAudioPlayer(sample_rate=22050, channels=1)

    try:
        start = time.perf_counter()
        stream = client.tts_stream(text)

        if play_audio and player.available():
            player.start()

        chunk_count = 0
        total_bytes = 0
        first_chunk_time = None

        print()
        print_header("Streaming Progress", 50)

        for chunk in stream:
            if chunk:
                now = time.perf_counter()
                if first_chunk_time is None:
                    first_chunk_time = now
                    ttfb = now - start
                    print_item("First Chunk", format_duration(ttfb))

                chunk_count += 1
                total_bytes += len(chunk)

                # Real-time playback
                if play_audio and player.started:
                    player.write_chunk(chunk)

                # Progress indicator every 5 chunks
                if chunk_count % 5 == 0:
                    elapsed = now - start
                    print_item(
                        "Progress",
                        f"Chunk {chunk_count} | {format_bytes(total_bytes)} | {format_duration(elapsed)}"
                    )

        elapsed = time.perf_counter() - start
        ttfb = first_chunk_time - start if first_chunk_time else 0

        # Calculate approximate audio duration from bytes (16-bit mono @ 22050Hz)
        audio_duration = total_bytes / (22050 * 2)
        rtf = elapsed / audio_duration if audio_duration > 0 else 0
        rtf_color = COLORS["green"] if rtf < 1.0 else COLORS["yellow"] if rtf < 2.0 else COLORS["reset"]

        print_item("Chunks", chunk_count)
        print_item("Total Size", format_bytes(total_bytes))
        print_item("TTFB", format_duration(ttfb))
        print_item("Total Time", format_duration(elapsed))
        print_item("Audio Duration", format_duration(audio_duration))
        print_item("RTF", f"{rtf_color}{rtf:.2f}{COLORS['reset']}")
        print_footer(50)
        print_stage("/tts/stream passed", "complete")

        player.stop()
        return True

    except Exception as e:
        player.stop()
        print_item("Error", str(e))
        print_footer(50)
        print_stage(f"/tts/stream failed: {e}", "failed")
        return False


# =============================================================================
# Interactive Mode
# =============================================================================

class InteractiveSession:
    """Interactive REPL for the TARS TTS client."""

    def __init__(self, client: TARSClient, play_audio: bool = True):
        self.client = client
        self.play_audio = play_audio
        self.use_streaming = True
        self.speed = 1.0
        self.temperature = 0.8
        self.top_p = 0.8
        self.top_k = 30
        self.running = True

    def print_help(self) -> None:
        """Print available commands."""
        print()
        print_header("Commands", 50)
        print_item("/help", "Show this help message")
        print_item("/quit, /exit", "Exit the client")
        print_item("/stream", "Toggle streaming mode (currently: " +
                   (f"{COLORS['green']}on{COLORS['reset']}" if self.use_streaming else f"{COLORS['dim']}off{COLORS['reset']}") + ")")
        print_item("/speed <0.5-2.0>", f"Set playback speed (currently: {self.speed})")
        print_item("/temp <0.1-1.5>", f"Set temperature (currently: {self.temperature})")
        print_item("/settings", "Show current settings")
        print_footer(50)
        print()

    def print_settings(self) -> None:
        """Print current settings."""
        print()
        print_header("Settings", 50)
        print_item("Streaming", f"{COLORS['green']}on{COLORS['reset']}" if self.use_streaming else f"{COLORS['dim']}off{COLORS['reset']}")
        print_item("Speed", f"{self.speed}")
        print_item("Temperature", f"{self.temperature}")
        print_item("Top P", f"{self.top_p}")
        print_item("Top K", f"{self.top_k}")
        print_item("Playback", f"{COLORS['green']}enabled{COLORS['reset']}" if self.play_audio else f"{COLORS['dim']}disabled{COLORS['reset']}")
        print_footer(50)
        print()

    def handle_command(self, cmd: str) -> bool:
        """Handle a slash command. Returns True if handled."""
        parts = cmd.strip().split(maxsplit=1)
        command = parts[0].lower()
        arg = parts[1] if len(parts) > 1 else None

        if command in ("/quit", "/exit", "/q"):
            self.running = False
            print(f"\n{COLORS['dim']}Goodbye!{COLORS['reset']}\n")
            return True

        elif command == "/help":
            self.print_help()
            return True

        elif command == "/stream":
            self.use_streaming = not self.use_streaming
            mode = f"{COLORS['green']}streaming{COLORS['reset']}" if self.use_streaming else f"{COLORS['cyan']}blocking{COLORS['reset']}"
            print_stage(f"Mode set to {mode}", "info")
            return True

        elif command == "/speed":
            if arg:
                try:
                    val = float(arg)
                    if 0.5 <= val <= 2.0:
                        self.speed = val
                        print_stage(f"Speed set to {val}", "info")
                    else:
                        print_stage("Speed must be between 0.5 and 2.0", "failed")
                except ValueError:
                    print_stage("Invalid speed value", "failed")
            else:
                print_stage(f"Current speed: {self.speed}", "info")
            return True

        elif command == "/temp":
            if arg:
                try:
                    val = float(arg)
                    if 0.1 <= val <= 1.5:
                        self.temperature = val
                        print_stage(f"Temperature set to {val}", "info")
                    else:
                        print_stage("Temperature must be between 0.1 and 1.5", "failed")
                except ValueError:
                    print_stage("Invalid temperature value", "failed")
            else:
                print_stage(f"Current temperature: {self.temperature}", "info")
            return True

        elif command == "/settings":
            self.print_settings()
            return True

        return False

    def synthesize(self, text: str) -> None:
        """Synthesize text to speech."""
        if self.use_streaming:
            self._synthesize_streaming(text)
        else:
            self._synthesize_blocking(text)

    def _synthesize_streaming(self, text: str) -> None:
        """Synthesize using streaming endpoint."""
        player = StreamingAudioPlayer(sample_rate=22050, channels=1)

        try:
            start = time.perf_counter()
            stream = self.client.tts_stream(
                text,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                speed=self.speed,
            )

            if self.play_audio and player.available():
                player.start()

            total_bytes = 0
            first_chunk = True

            for chunk in stream:
                if chunk:
                    if first_chunk:
                        ttfb = time.perf_counter() - start
                        print_stage(f"First chunk in {format_duration(ttfb)}", "info")
                        first_chunk = False

                    total_bytes += len(chunk)
                    if self.play_audio and player.started:
                        player.write_chunk(chunk)

            elapsed = time.perf_counter() - start
            audio_duration = total_bytes / (22050 * 2)
            print_stage(f"Done ({format_bytes(total_bytes)}, {format_duration(audio_duration)} audio, {format_duration(elapsed)} total)", "complete")

        except Exception as e:
            print_stage(f"Synthesis failed: {e}", "failed")
        finally:
            player.stop()

    def _synthesize_blocking(self, text: str) -> None:
        """Synthesize using blocking endpoint."""
        try:
            start = time.perf_counter()
            print_stage("Generating...", "progress")

            audio_bytes = self.client.tts(
                text,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                speed=self.speed,
            )

            elapsed = time.perf_counter() - start
            print_stage(f"Generated {format_bytes(len(audio_bytes))} in {format_duration(elapsed)}", "complete")

            if self.play_audio:
                play_audio_bytes(audio_bytes, is_wav=True)

        except Exception as e:
            print_stage(f"Synthesis failed: {e}", "failed")

    def run(self) -> None:
        """Run the interactive REPL."""
        # Try to enable readline for history
        try:
            import readline  # noqa: F401
        except ImportError:
            pass

        self.print_help()
        prompt = f"{COLORS['cyan']}TARS>{COLORS['reset']} "

        while self.running:
            try:
                text = input(prompt).strip()

                if not text:
                    continue

                if text.startswith("/"):
                    if not self.handle_command(text):
                        print_stage(f"Unknown command: {text.split()[0]}. Type /help for commands.", "failed")
                else:
                    self.synthesize(text)

            except KeyboardInterrupt:
                print(f"\n{COLORS['dim']}Use /quit to exit{COLORS['reset']}")
            except EOFError:
                self.running = False
                print(f"\n{COLORS['dim']}Goodbye!{COLORS['reset']}\n")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="TARS TTS Server Client - Comprehensive endpoint testing with audio playback",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                          # Run all tests with default text
  %(prog)s -i                       # Interactive mode (REPL)
  %(prog)s --text "Hello world"     # Use custom text
  %(prog)s --stream-only            # Only test streaming endpoint
  %(prog)s --no-playback            # Skip audio playback
  %(prog)s --url http://host:8009   # Use custom server URL
        """,
    )
    parser.add_argument("--url", default="http://localhost:8009", help="Server base URL")
    parser.add_argument("-i", "--interactive", action="store_true", help="Interactive REPL mode")
    parser.add_argument("--text", default=None, help="Custom text to synthesize")
    parser.add_argument("--stream-only", action="store_true", help="Only test /tts/stream endpoint")
    parser.add_argument("--no-playback", action="store_true", help="Skip audio playback")
    parser.add_argument("--no-stream", action="store_true", help="Skip streaming test")

    args = parser.parse_args()

    # Client setup
    client = TARSClient(args.url)
    play_audio = not args.no_playback

    # Print banner
    print()
    print_header("TARS TTS Client", 60)
    print_item("Server", args.url)
    print_item("Mode", f"{COLORS['magenta']}interactive{COLORS['reset']}" if args.interactive else f"{COLORS['cyan']}test{COLORS['reset']}")
    print_item("Playback", f"{COLORS['green']}enabled{COLORS['reset']}" if play_audio else f"{COLORS['dim']}disabled{COLORS['reset']}")
    print_item("Time", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    print_footer(60)

    # Wait for server
    print()
    print_stage("Connecting to server...", "progress")
    if not client.wait_for_ready(max_retries=5, retry_delay=2.0):
        print_stage("Server unreachable. Is 'uv run python serve_tars.py' running?", "failed")
        sys.exit(1)
    print_stage("Server connected", "complete")

    # Interactive mode
    if args.interactive:
        session = InteractiveSession(client, play_audio=play_audio)
        session.run()
        sys.exit(0)

    # Test mode
    blocking_text = args.text or "Hello, this is a test of the TARS voice synthesis server. The quick brown fox jumps over the lazy dog."
    stream_text = args.text or "This is a longer streaming test. We want to ensure that chunks arrive smoothly and audio plays back in real time as data is received from the server."

    all_passed = True

    if not args.stream_only:
        if not test_healthz(client):
            all_passed = False
        if not test_readyz(client):
            all_passed = False
        if not test_tts(client, blocking_text, play_audio=play_audio):
            all_passed = False

    if not args.no_stream:
        if not test_tts_stream(client, stream_text, play_audio=play_audio):
            all_passed = False

    # Summary
    print()
    print_header("Summary", 40)
    if all_passed:
        print_item("Result", f"{COLORS['green']}All tests passed{COLORS['reset']}")
    else:
        print_item("Result", f"{COLORS['red']}Some tests failed{COLORS['reset']}")
    print_footer(40)
    print()

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()
