###########################################################################################
#  agents process to Simulate a debate between a socialits and a capitalist               #
#                               thank you!                                                #
#                                                                                         #
###########################################################################################


import dearpygui.dearpygui as dpg
import threading
import queue
import random
from crewai import Agent, Task, Crew, LLM
import os
import logging
import sys
import time
import builtins
import traceback
import re
from markdown_utils import format_section_header, format_message, format_conclusion, generate_debate_metadata

class DebateTranscript:
    def __init__(self, file_path):
        self.file_path = file_path
        self.sections = []  # Sections to be added to markdown file
        self.topic = None  # Debate topic
        self.content = []  # Store content here (raw format)

    def add_section(self, role, message):
        """Add a debate section with role and message."""
        section_header = format_section_header(role)
        formatted_message = format_message(message)
        self.sections.append(section_header + formatted_message)  # Add formatted section
        self.content.append(f"{role}: {message}")  # Add raw content

    def add_system_message(self, message):
        """Add a system message."""
        section_header = format_section_header("System Message")
        formatted_message = format_message(message)
        self.sections.append(section_header + formatted_message)  # Add system message section
        self.content.append(f"System: {message}")  # Add raw content

    def add_conclusion(self, result):
        """Add the conclusion of the debate."""
        # If result is a CrewOutput object, try to extract meaningful information
        if hasattr(result, 'get_final_output'):
            conclusion_text = result.get_final_output()
        elif hasattr(result, '__str__'):
            conclusion_text = str(result)
        else:
            conclusion_text = "Debate concluded without a specific conclusion."
        
        conclusion = format_conclusion(conclusion_text)
        self.sections.append(conclusion)  # Add formatted conclusion
        self.content.append(f"Conclusion: {conclusion_text}")  # Add raw content

    def save(self):
        """Save the debate transcript to a markdown file."""
        try:
            # Add metadata generation before writing sections
            metadata = generate_debate_metadata(self.topic)
            directory = os.path.dirname(self.file_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)

            with open(self.file_path, 'w') as file:
                # Write metadata
                file.write(metadata)
                
                # Write topic header
                file.write(f"# Debate Topic: {self.topic}\n\n")
                
                # Write formatted debate sections
                file.write("\n".join(self.sections))

                # Add a section for raw transcript
                file.write("\n\n## Full Transcript\n\n")
                file.write("```\n")
                file.write("\n".join(self.content))
                file.write("\n```")

            print(f"Debate transcript saved successfully at {self.file_path}")
        except Exception as e:
            print(f"Error saving debate transcript: {e}")

class StreamingPrintCapture:
    def __init__(self, message_queue, debate_transcript):
        self.message_queue = message_queue
        self.debate_transcript = debate_transcript
        self.capture_buffer = ""
        self.lock = threading.Lock()
        self.terminal = sys.stdout  # Keep the original terminal output

    def write(self, text):
        with self.lock:
            # Log to the terminal
            self.terminal.write(text)
            self.terminal.flush()  # Ensure messages appear immediately in the terminal

            # Process for GUI
            self.capture_buffer += text
            lines = self.capture_buffer.split('\n')
            for line in lines[:-1]:  # Process complete lines
                self.process_line(line)
            self.capture_buffer = lines[-1]  # Keep the last (incomplete) line in buffer

    def process_line(self, line):
        line = re.sub(r'\x1b\[[0-9;]*m', '', line).strip()  # Clean ANSI codes
        if not line:
            return

        try:
            if '[MODERATOR]' in line:
                clean_line = line.replace('[MODERATOR]', '').strip()
                self.message_queue.put((f"[Moderator] {clean_line}", (0, 0, 255)))
                self.debate_transcript.add_section('Moderator', clean_line)
            elif '[SOCIALIST]' in line:
                clean_line = line.replace('[SOCIALIST]', '').strip()
                self.message_queue.put((f"[Socialist] {clean_line}", (0, 255, 0)))
                self.debate_transcript.add_section('Socialist', clean_line)
            elif '[CAPITALIST]' in line:
                clean_line = line.replace('[CAPITALIST]', '').strip()
                self.message_queue.put((f"[Capitalist] {clean_line}", (255, 0, 0)))
                self.debate_transcript.add_section('Capitalist', clean_line)
            elif '[SYSTEM]' in line or 'Finished' in line:
                self.message_queue.put((line, (128, 128, 128)))
                self.debate_transcript.add_system_message(line)
        except Exception as e:
            logging.error(f"Error processing line: {e}")

    def flush(self):
        self.terminal.flush()  # Ensure flush works for both terminal and GUI


class DebateGUIApp:
    def __init__(self, topic=None, num_rounds=3):
        self.is_debate_running = threading.Event()
        self.message_queue = queue.Queue()
        self.status_message = threading.Event()
        self.topic_list = [
            "Universal Basic Income",
            "Climate Change Economic Policies", 
            "Minimum Wage Legislation", 
            "Tech Monopolies and Regulation",
            "Green Energy Transition",
            "Wealth Redistribution",
            "Global Trade Policies"
        ]
        self.topic = topic or random.choice(self.topic_list)
        self.num_rounds = num_rounds
        self.is_debate_running = threading.Event()
        self.markdown_file_path = os.path.join(os.path.expanduser("~"), "Laboratories", "_ai", "src", "debate_agents", "oooooo_debate_output.md")
        
        # Initialize DebateTranscript
        self.debate_transcript = DebateTranscript(self.markdown_file_path)
        self.debate_transcript.topic = self.topic

        # LLM models configuration
        self.llm_models = {
            "Moderator": ["ollama/moderator-gemma2-9:latest", "ollama/default-moderator"],
            "Socialist": ["ollama/socialism-gemma2-9:latest", "ollama/alternative-socialist"],
            "Capitalist": ["ollama/capitalism-gemma2-9:latest", "ollama/alternative-capitalist"]
        }
        
        # Selected LLM models (can be changed via UI)
        self.selected_models = {
            "Moderator": self.llm_models["Moderator"][0],
            "Socialist": self.llm_models["Socialist"][0],
            "Capitalist": self.llm_models["Capitalist"][0]
        }

        # LLM configuration
        self.llm_config = {
            "base_url": "http://localhost:11434",
            "temperature": 0.7
        }

    def create_debate_agents(self, topic):
        """Create debate agents with dynamically selected LLM models"""
        def create_llm(model_name):
            return LLM(
                model=model_name,
                temperature=self.llm_config["temperature"],
                base_url=self.llm_config["base_url"]
            )

        # Moderator Agent
        moderator_agent = Agent(
            role='Expert Debate Moderator',
            goal=f'Facilitate a structured and balanced discussion on {topic}, ensuring fair representation of different perspectives.',
            backstory=f'You are an impartial moderator committed to maintaining a respectful and constructive dialogue about {topic}. '
                      f'Your role is to introduce debate rounds, ask clarifying questions, summarize key points, and ensure each perspective '
                      f'has an equal opportunity to present their arguments.',
            verbose=True,
            allow_delegation=False,
            llm=create_llm(self.selected_models["Moderator"])
        )

        # Socialist Perspective Agent
        socialist_agent = Agent(
            role='Socialist Economic Specialist',
            goal=f'Analyze {topic} through a socialist economic lens.',
            backstory=f'You are a socialist economic theorist who views {topic} as a potential tool for social transformation. '
                      f'Your analysis focuses on how {topic} can address systemic inequalities, provide a social safety net, and potentially restructure '
                      f'economic relationships. You see {topic} as more than just a financial policy - it\'s a step towards economic democracy and collective well-being. '
                      f'You adapt your argumentation style based on debate dynamics.',
            verbose=True,
            allow_delegation=False,
            llm=create_llm(self.selected_models["Socialist"])
        )

        # Capitalist Perspective Agent
        capitalist_agent = Agent(
            role='Capitalist Economic Specialist',
            goal=f'Evaluate {topic} through a free-market economic perspective.',
            backstory=f'You are a market-oriented economic analyst who approaches {topic} with a focus on economic efficiency, '
                      f'individual incentives, and potential market implications. Your analysis will examine how {topic} might interact with labor markets, '
                      f'entrepreneurship, and economic productivity. '
                      f'You modify your arguments responsively to the debate flow.',
            verbose=True,
            allow_delegation=False,
            llm=create_llm(self.selected_models["Capitalist"])
        )

        return moderator_agent, socialist_agent, capitalist_agent

    def create_chat_debate_tasks(self, moderator, socialist, capitalist, topic, num_rounds=3):

        debate_tasks = []

        # Opening moderation task
        debate_tasks.append(
            Task(
                description=f'Debate Introduction: Set the stage for a comprehensive discussion on {topic}. '
                            f'Provide context, establish ground rules for respectful dialogue, and introduce the participants. '
                            f'Frame the debate as an exploration of different economic perspectives.',
                agent=moderator,
                expected_output=f'A structured introduction to the {topic} debate, setting a tone of mutual respect and intellectual curiosity.'
            )
        )

        # Iterative debate rounds (similar to your original implementation)
        for round_num in range(1, num_rounds + 1):
            # Socialist's opening argument
            debate_tasks.append(
                Task(
                    description=f'Round {round_num} Opening (Socialist Perspective): Present a comprehensive 3-paragraph argument '
                                f'exploring {topic} through a socialist economic framework. Highlight potential social benefits, '
                                'systemic transformations, and strategies for implementing economic equity.',
                    agent=socialist,
                    expected_output=f'A detailed socialist analysis of {topic}, emphasizing collective welfare and social justice.'
                )
            )

            # Moderator's response and question
            debate_tasks.append(
                Task(
                    description=f'Moderator Reflection (Round {round_num}): Summarize the key points of the socialist argument. '
                                f'Pose a challenging but fair question that invites the capitalist perspective to provide deeper insights '
                                f'or address potential counterarguments about {topic}.',
                    agent=moderator,
                    expected_output=f'A balanced summary and thought-provoking question to advance the {topic} debate.'
                )
            )

            # Capitalist's response
            debate_tasks.append(
                Task(
                    description=f'Round {round_num} Response (Capitalist Perspective): Address the moderator\'s question and the previous '
                                f'socialist argument. Develop a 3-paragraph response that critically examines {topic} from a free-market '
                                'economic standpoint, focusing on efficiency, innovation, and individual economic freedoms.',
                    agent=capitalist,
                    expected_output=f'A strategic analysis critiquing the socialist approach to {topic}.'
                )
            )

            # Moderator's closing remarks for the round
            debate_tasks.append(
                Task(
                    description=f'Moderator Round Summary (Round {round_num}): Provide a neutral synthesis of the arguments presented. '
                                f'Highlight key points of agreement and divergence in the discussion about {topic}. Prepare the participants '
                                'for the next round of debate.',
                    agent=moderator,
                    expected_output=f'A balanced summary of the arguments in Round {round_num} of the {topic} debate.'
                )
            )

        # Final moderation task
        debate_tasks.append(
            Task(
                description=f'Debate Conclusion: Synthesize the key insights from the entire debate on {topic}. '
                            f'Reflect on the complexity of the economic perspectives shared and invite final thoughts from both participants.',
                agent=moderator,
                expected_output=f'A comprehensive conclusion that captures the nuanced perspectives on {topic}.'
            )
        )

        return debate_tasks


    def run_debate(self):
        """
        Run debate in a separate thread and stream messages to the queue.
        """
        # Create a new transcript with the current topic
        self.debate_transcript = DebateTranscript(self.markdown_file_path)
        self.debate_transcript.topic = self.topic

        # Redirect stdout to our custom capture
        stream_capture = StreamingPrintCapture(self.message_queue, self.debate_transcript)
        original_stdout = sys.stdout
        sys.stdout = stream_capture

        try:
            # Create agents
            moderator, socialist, capitalist = self.create_debate_agents(self.topic)

            # Modify agent communication to use custom print
            moderator.backstory = "[MODERATOR] " + moderator.backstory
            socialist.backstory = "[SOCIALIST] " + socialist.backstory
            capitalist.backstory = "[CAPITALIST] " + capitalist.backstory

            # Create tasks and start debate
            debate_tasks = self.create_chat_debate_tasks(
                moderator, socialist, capitalist, self.topic, self.num_rounds)

            # Use print statements with specific tags for better streaming
            print("[SYSTEM] Starting debate...")
            print(f"[SYSTEM] Debate Topic: {self.topic}")
            
            crew = Crew(
                agents=[moderator, socialist, capitalist],
                tasks=debate_tasks,
                verbose=True,  # Ensure verbose mode for more streaming messages
                embedder={
                    "provider": "ollama",
                    "config": {
                        "model": "nomic-embed-text"
                    }
                },
                output_log_file="ooooooooo_crew_debate_result.md"
            )
            
            result = crew.kickoff()

            # Add conclusion to transcript
            self.debate_transcript.add_conclusion(result)

            # Final system messages
            print("[SYSTEM] Processing debate conclusion...")
            print("[SYSTEM] Debate Completed.")

        except Exception as e:
            error_msg = f"[SYSTEM] Debate Error: {str(e)}\n{traceback.format_exc()}"
            print(error_msg)
        
        finally:
            # Restore stdout
            sys.stdout = original_stdout
            
            # Save transcript
            self.debate_transcript.save()
            
            self.is_debate_running.clear()
    
    def check_messages(self):
        """Check and process messages from the queue with real-time streaming."""
        try:
            messages_processed = 0
            max_messages_per_frame = 15  # Limit to prevent blocking

            # Process messages from the queue
            while not self.message_queue.empty() and messages_processed < max_messages_per_frame:
                try:
                    message, color = self.message_queue.get_nowait()
                    dpg.add_text(message, parent="debate_window", color=color)
                    messages_processed += 1
                except queue.Empty:
                    break

            # Only mark debate as finished if it's not running AND there are no remaining messages
            if not self.is_debate_running.is_set() and self.message_queue.empty():
                dpg.add_text("[SYSTEM] Debate Finished", parent="debate_window", color=(128, 128, 128))
                dpg.set_value("status_bar", "Debate Completed")

        except Exception as e:
            print(f"Error processing messages: {e}")

    def run_threaded_debate(self):
        """Initiate debate in a thread with status updates."""
        # Update status
        dpg.set_value("status_bar", f"Debate Started: {self.topic}")
        
        # Reset windows
        dpg.delete_item("debate_window", children_only=True)
        dpg.add_text("[SYSTEM] Debate Starting...", parent="debate_window", color=(128, 128, 128))
        
        # Start debate thread
        self.is_debate_running.set()
        debate_thread = threading.Thread(target=self.run_debate, daemon=True)
        debate_thread.start()

    def create_custom_theme(self):
        """Create a dark-themed custom UI theme."""
        with dpg.theme() as global_theme:
            with dpg.theme_component(dpg.mvAll):
                # Global text color
                dpg.add_theme_color(dpg.mvThemeCol_Text, (200, 200, 200))
            
            # Specific component theming
            with dpg.theme_component(dpg.mvChildWindow):
                dpg.add_theme_color(dpg.mvThemeCol_ChildBg, (30, 30, 30))
                dpg.add_theme_color(dpg.mvThemeCol_Border, (70, 70, 70))
            
            with dpg.theme_component(dpg.mvButton):
                dpg.add_theme_color(dpg.mvThemeCol_Button, (50, 50, 50))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonHovered, (70, 70, 70))
                dpg.add_theme_color(dpg.mvThemeCol_ButtonActive, (90, 90, 90))
        
        return global_theme

    def setup_configuration_window(self):
        """Create a configuration window for debate settings."""
        with dpg.window(tag="config_window", label="Debate Configuration", width=400, height=300, pos=(850, 50)):
            # Topic Selection
            dpg.add_text("Select Debate Topic:")
            with dpg.group(horizontal=False):
                topic_combo = dpg.add_combo(tag="topic_combo", default_value=self.topic, width=-1, items=self.topic_list)
            
            # Rounds Selection
            dpg.add_text("Number of Debate Rounds:")
            dpg.add_slider_int(tag="rounds_slider", default_value=self.num_rounds, min_value=1, max_value=5, width=-1)
            
            # LLM Model Selections
            dpg.add_text("Select LLM Models:")
            for role in ["Moderator", "Socialist", "Capitalist"]:
                dpg.add_text(f"{role} Model:")
                with dpg.group(horizontal=False):
                    model_combo = dpg.add_combo(
                        tag=f"{role.lower()}_model_combo", 
                        default_value=self.selected_models[role], 
                        width=-1, 
                        items=self.llm_models[role]
                    )
            
            # Apply Configuration Button
            dpg.add_button(label="Apply Configuration", callback=self.apply_configuration)

    def apply_configuration(self):
        """Apply the selected configuration."""
        # Update topic
        self.topic = dpg.get_value("topic_combo")
        
        # Update rounds
        self.num_rounds = dpg.get_value("rounds_slider")
        
        # Update LLM models
        self.selected_models["Moderator"] = dpg.get_value("moderator_model_combo")
        self.selected_models["Socialist"] = dpg.get_value("socialist_model_combo")
        self.selected_models["Capitalist"] = dpg.get_value("capitalist_model_combo")
        
        # Update topic display in the main window
        dpg.set_value("topic_display", f"Current Topic: {self.topic}")
        
        # Update status bar
        dpg.set_value("status_bar", f"Configuration Updated: {self.topic}, {self.num_rounds} rounds")

    def primary_window(self):
        """Set up the primary debate window."""
        with dpg.window(tag="main_window", label="Economic Debate Simulator", width=800, height=600):
            # Toolbar with action buttons
            with dpg.group(horizontal=True):
                dpg.add_button(label="Start Debate", callback=self.run_threaded_debate)
                dpg.add_button(label="Pause Debate", callback=self.pause_debate)
                dpg.add_button(label="Save Transcript", callback=self.save_transcript)

            # Debate topic display
            dpg.add_text(f"Current Topic: {self.topic}", tag="topic_display")

            # Debate transcript window
            with dpg.child_window(tag="debate_window", width=-1, height=450):
                pass

            # Status bar
            with dpg.group(horizontal=True):
                dpg.add_text("Status:", color=(100, 200, 100))
                dpg.add_text("Ready", tag="status_bar", color=(200, 200, 200))
                
    def start_debate(self):
        dpg.create_context()

        # Setup GUI components
        global_theme = self.create_custom_theme()
        self.primary_window()
        self.setup_configuration_window()

        # Bind theme
        dpg.bind_theme(global_theme)

        # Setup viewport
        dpg.create_viewport(title="Economic Debate Simulator", width=1280, height=720)
        dpg.show_viewport()

        # Regularly check for messages
        dpg.set_frame_callback(1, self.check_messages)  # Runs every frame

        dpg.setup_dearpygui()
        dpg.start_dearpygui()
        dpg.destroy_context()

    
    def pause_debate(self):
        """Pause or resume the ongoing debate."""
        if self.is_debate_running.is_set():
            self.is_debate_running.clear()
            dpg.set_value("status_bar", "Debate Paused")
        else:
            self.is_debate_running.set()
            dpg.set_value("status_bar", f"Debate Resumed: {self.topic}")

    def save_transcript(self):
        """Manually trigger transcript saving."""
        try:
            self.debate_transcript.save()
            dpg.set_value("status_bar", f"Transcript Saved: {self.markdown_file_path}")
        except Exception as e:
            dpg.set_value("status_bar", f"Error Saving Transcript: {str(e)}")

    def show_error_modal(self, error_message):
        """Display an error modal with details."""
        with dpg.window(label="Debate Error", modal=True, width=400):
            dpg.add_text(error_message)
            dpg.add_button(label="Close", callback=lambda: dpg.delete_item(dpg.last_container()))

class QueueLogger:
    def __init__(self, message_queue, color):
        self.message_queue = message_queue
        self.color = color

    def write(self, message):
        if message.strip():  # Avoid empty logs
            self.message_queue.put((message, self.color))

    def flush(self):
        pass  # Required for Python's file-like objects

def main():
    debate_app = DebateGUIApp()
    debate_app.start_debate()

if __name__ == "__main__":
    main()