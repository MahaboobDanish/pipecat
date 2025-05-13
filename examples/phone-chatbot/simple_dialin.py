#
# Copyright (c) 2024–2025, Daily
#
# SPDX-License-Identifier: BSD 2-Clause License
#
import argparse
import asyncio
import os
import sys
import json

from call_connection_manager import CallConfigManager, SessionManager
from dotenv import load_dotenv
from loguru import logger

from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.vad.silero import SileroVADAnalyzer
from pipecat.frames.frames import EndTaskFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.processors.frame_processor import FrameDirection
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.llm_service import FunctionCallParams
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.services.daily import DailyDialinSettings, DailyParams, DailyTransport

# Add these new imports at the top (Part of silence detection infrastructure)
from datetime import datetime
from pipecat.frames.frames import TextFrame

load_dotenv(override=True)

logger.remove(0)
logger.add(sys.stderr, level="DEBUG")

daily_api_key = os.getenv("DAILY_API_KEY", "")
daily_api_url = os.getenv("DAILY_API_URL", "https://api.daily.co/v1")

# Constants for configuration (Implements silence timeout and prompt thresholds)
SILENCE_TIMEOUT = 10  # seconds
MAX_UNANSWERED_PROMPTS = 3
CHECK_INTERVAL = 1  # second between silence checks

class SilenceMonitor:
    def __init__(self, session_manager, tts_service, llm_service):
        # Stores call session state (Part of state management)
        self.session_manager = session_manager
        # TTS service for prompts (Implements TTS prompt playback)
        self.tts = tts_service  
        # LLM service for message queuing (Handles termination)
        self.llm = llm_service
        # Timing and counters (Core silence detection logic)
        self.last_activity_time = datetime.now()
        self.unanswered_prompts = 0
        self.active = False
    
    async def check_silence(self):
        if not self.active or self.session_manager.call_flow_state.call_terminated:
            return  # Exit early if monitoring is disabled or call already terminated

        silence_duration = (datetime.now() - self.last_activity_time).total_seconds()
        logger.debug(f"Silence check: {silence_duration:.1f}s of silence")

        if silence_duration > SILENCE_TIMEOUT:
            # Prevent multiple increments if we're already over threshold
            if self.unanswered_prompts >= MAX_UNANSWERED_PROMPTS:
                return  # Already reached max prompts
            
            self.unanswered_prompts += 1
            self.session_manager.call_flow_state.silence_events += 1
            
            logger.debug(f"Unanswered prompts: {self.unanswered_prompts}/{MAX_UNANSWERED_PROMPTS}")
            self.session_manager.call_flow_state.unanswered_prompts = self.unanswered_prompts
            if self.unanswered_prompts >= MAX_UNANSWERED_PROMPTS:
                logger.info("Max unanswered prompts reached - terminating call")
                await self.terminate_call()
                return  # Critical - exit after termination
                
            try:
                # Play TTS prompt with cooldown
                self.prompt = "Are you still there? Please let me know how I can help."
                logger.info(f"Playing silence prompt # {self.prompt}")
                
                # Queue prompt through LLM and TTS
                await self.llm.queue_frame(TextFrame(self.prompt))
                # await self.tts.run_tts(self.prompt)
                
                # # Add cooldown period after prompt (prevents immediate re-trigger)
                # self.last_activity_time = datetime.now() + timedelta(seconds=5)
                
            except Exception as e:
                logger.error(f"Failed to queue TTS prompt: {e}")
                self.reset_timer()

    
        
    def reset_timer(self):
        """Resets detection timers on user activity"""
        self.last_activity_time = datetime.now()
        
    def start_monitoring(self):
        """Enables monitoring when call becomes active"""
        self.active = True
        self.reset_timer()
        # Start periodic checking (Enables continuous monitoring)
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        # asyncio.create_task(self._monitor_loop())
        
    async def _monitor_loop(self):
        """Background task for periodic silence checks"""
        while self.active:
            await self.check_silence()
            await asyncio.sleep(CHECK_INTERVAL)


    async def terminate_call(self):
        """Handles call termination flow"""
        # Prevent multiple termination attempts
        if self.session_manager.call_flow_state.call_terminated:
            return
            
        logger.info("Max unanswered prompts reached - terminating call")
        
        # 1. Finalize call stats first
        self.session_manager.call_flow_state.finalize_call()
        self.session_manager.call_flow_state.termination_reason = \
            f"Terminated after {MAX_UNANSWERED_PROMPTS} unanswered prompts"
        
        # 2. Log summary before any cleanup
        await self._log_summary()
        
        # 3. Stop monitoring
        self.active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            
        # 4. Initiate termination
        await self.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    async def _log_summary(self):
        """Log call summary if not already logged"""
        if not self.session_manager.call_flow_state.summary_finished:
            cf_state = self.session_manager.call_flow_state
            cf_state.end_time = datetime.now()  # Ensure end_time is set
            
            summary = {
                "duration_seconds": round(cf_state.duration, 2),
                "silence_events": cf_state.silence_events,
                "unanswered_prompts": cf_state.unanswered_prompts,
                "termination_reason": cf_state.termination_reason
            }
            
            logger.info(f"Call summary: {json.dumps(summary, indent=2)}")
            cf_state.summary_finished = True

async def main(
    room_url: str,
    token: str,
    body: dict,
):
    # ------------ CONFIGURATION AND SETUP ------------

    # Create a config manager using the provided body
    call_config_manager = CallConfigManager.from_json_string(body) if body else CallConfigManager()

    # Get important configuration values
    test_mode = call_config_manager.is_test_mode()

    # Get dialin settings if present
    dialin_settings = call_config_manager.get_dialin_settings()

    # Initialize the session manager
    session_manager = SessionManager()

    # ------------ TRANSPORT SETUP ------------

    # Set up transport parameters
    if test_mode:
        logger.info("Running in test mode")
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )
    else:
        daily_dialin_settings = DailyDialinSettings(
            call_id=dialin_settings.get("call_id"), call_domain=dialin_settings.get("call_domain")
        )
        transport_params = DailyParams(
            api_url=daily_api_url,
            api_key=daily_api_key,
            dialin_settings=daily_dialin_settings,
            audio_in_enabled=True,
            audio_out_enabled=True,
            video_out_enabled=False,
            vad_analyzer=SileroVADAnalyzer(),
            transcription_enabled=True,
        )

    # Initialize transport with Daily
    transport = DailyTransport(
        room_url,
        token,
        "Simple Dial-in Bot",
        transport_params,
    )

    # Initialize TTS
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY", ""),
        voice_id="b7d50908-b17c-442d-ad8d-810c63997ed9",  # Use Helpful Woman voice by default
    )

    # ------------ FUNCTION DEFINITIONS ------------

    async def terminate_call(params: FunctionCallParams):
        """Function the bot can call to terminate the call upon completion of a voicemail message."""
        if session_manager:
            # Mark that the call was terminated by the bot
            session_manager.call_flow_state.set_call_terminated()

        # Then end the call
        await params.llm.queue_frame(EndTaskFrame(), FrameDirection.UPSTREAM)

    # Define function schemas for tools
    terminate_call_function = FunctionSchema(
        name="terminate_call",
        description="Call this function to terminate the call.",
        properties={},
        required=[],
    )

    # Create tools schema
    tools = ToolsSchema(standard_tools=[terminate_call_function])

    # ------------ LLM AND CONTEXT SETUP ------------

    # Set up the system instruction for the LLM
    system_instruction = """You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself. If the user ends the conversation, **IMMEDIATELY** call the `terminate_call` function. """

    # Initialize LLM
    llm = OpenAILLMService(api_key=os.getenv("OPENAI_API_KEY"))

    # Register functions with the LLM
    llm.register_function("terminate_call", terminate_call)

    # As the LLM Context and Message is used in our Silence Monitor initialize the SilenceMonitor here
    silence_monitor = SilenceMonitor(session_manager, tts, llm)
    # silence_monitor.start_monitoring()

    # Create system message and initialize messages list
    messages = [call_config_manager.create_system_message(system_instruction)]

    # Initialize LLM context and aggregator
    context = OpenAILLMContext(messages, tools)
    context_aggregator = llm.create_context_aggregator(context)

    # ------------ PIPELINE SETUP ------------

    # Build pipeline
    pipeline = Pipeline(
        [
            transport.input(),  # Transport user input
            context_aggregator.user(),  # User responses
            llm,  # LLM
            tts,  # TTS
            transport.output(),  # Transport bot output
            context_aggregator.assistant(),  # Assistant spoken responses
        ]
    )

    # Create pipeline task
    task = PipelineTask(pipeline, params=PipelineParams(allow_interruptions=True))

    # ------------ EVENT HANDLERS ------------

    @transport.event_handler("on_first_participant_joined")
    async def on_first_participant_joined(transport, participant):
        session_manager.call_flow_state.start_time = datetime.now()
        logger.debug(f"First participant joined: {participant['id']}")
        await transport.capture_participant_transcription(participant["id"])
        await task.queue_frames([context_aggregator.user().get_context_frame()])
        # start the monitoring as participant joined the call
        silence_monitor.start_monitoring()

    # Events to detect the speech
    @transport.event_handler("on_dialout_answered")
    async def on_speech_start(transport,frame):
        silence_monitor.reset_timer()

    @transport.event_handler("on_dialout_stopped")
    async def on_speech_stop(transport, frame):
        silence_monitor.reset_timer()

    
    @transport.event_handler("on_participant_left")
    async def on_participant_left(transport, participant, reason):
        cf_state = session_manager.call_flow_state
        if not cf_state.summary_finished:
            cf_state.termination_reason = f"Participant left: {reason}"
            cf_state.finalize_call()
            await silence_monitor._log_summary()
        await task.cancel()

    # ------------ RUN PIPELINE ------------

    if test_mode:
        logger.debug("Running in test mode (can be tested in Daily Prebuilt)")

    runner = PipelineRunner()
    await runner.run(task)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple Dial-in Bot")
    parser.add_argument("-u", "--url", type=str, help="Room URL")
    parser.add_argument("-t", "--token", type=str, help="Room Token")
    parser.add_argument("-b", "--body", type=str, help="JSON configuration string")

    args = parser.parse_args()

    # Log the arguments for debugging
    logger.info(f"Room URL: {args.url}")
    logger.info(f"Token: {args.token}")
    logger.info(f"Body provided: {bool(args.body)}")

    asyncio.run(main(args.url, args.token, args.body))
