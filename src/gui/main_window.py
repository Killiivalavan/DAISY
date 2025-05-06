"""
Main window implementation for the JARVIS-inspired GUI.
"""
import sys
import os
import math
from datetime import datetime
import numpy as np
from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QFrame, QPushButton, QStatusBar)
from PyQt6.QtCore import Qt, QTimer, QSize, QPoint, QPointF, QRectF, pyqtSignal, pyqtSlot
from PyQt6.QtGui import (QPainter, QColor, QPen, QBrush, QFont, QFontDatabase,
                        QPainterPath, QLinearGradient, QRadialGradient, QPaintEvent)
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

class AudioVisualizer(QWidget):
    """Audio visualization component similar to JARVIS interface."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(300, 300)
        
        # Animation properties
        self.rotation = 0
        self.inner_rotation = 0
        self.pulse_size = 0
        self.pulse_direction = 1
        
        # State indicators
        self.is_listening = False
        self.is_processing = False
        self.is_speaking = False
        
        # Audio data
        self.audio_data = np.zeros(128)
        
        # Animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start(30)  # ~30fps
        
        # Color scheme inspired by JARVIS UI
        self.primary_color = QColor(0, 180, 220)
        self.secondary_color = QColor(0, 140, 180)
        self.dark_color = QColor(10, 20, 40)
        self.highlight_color = QColor(0, 220, 255)
        self.background_color = QColor(0, 10, 20)
        
    def update_animation(self):
        """Update animation parameters."""
        self.rotation = (self.rotation + 0.5) % 360
        self.inner_rotation = (self.inner_rotation - 0.3) % 360
        
        # Pulse effect
        self.pulse_size += 0.05 * self.pulse_direction
        if self.pulse_size > 1.0:
            self.pulse_direction = -1
        elif self.pulse_size < 0.2:
            self.pulse_direction = 1
            
        self.update()
        
    def update_audio_data(self, data):
        """Update the audio visualization data."""
        self.audio_data = data
        self.update()
        
    def set_state(self, listening=False, processing=False, speaking=False):
        """Set the current state of the assistant."""
        self.is_listening = listening
        self.is_processing = processing
        self.is_speaking = speaking
        self.update()
        
    def paintEvent(self, event: QPaintEvent):
        """Draw the JARVIS-like interface."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Setup
        center = QPoint(self.width() // 2, self.height() // 2)
        center_x = self.width() / 2
        center_y = self.height() / 2
        min_dim = min(self.width(), self.height())
        outer_radius = min_dim * 0.45
        inner_radius = min_dim * 0.3
        
        # Background
        painter.fillRect(self.rect(), self.background_color)
        
        # Draw outer ring - convert float radii to integers for QPoint
        pen = QPen(self.primary_color, 2)
        painter.setPen(pen)
        painter.drawEllipse(center, int(outer_radius), int(outer_radius))
        
        # Draw inner ring with gradient - use float coordinates directly
        gradient = QRadialGradient(center_x, center_y, inner_radius)
        gradient.setColorAt(0.7, QColor(0, 180, 220, 30))
        gradient.setColorAt(1.0, QColor(0, 180, 220, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(center, int(inner_radius), int(inner_radius))
        
        # Draw rotating segments on outer ring
        painter.save()
        painter.translate(center)
        painter.rotate(self.rotation)
        
        segment_count = 36
        segment_angle = 360 / segment_count
        for i in range(segment_count):
            if i % 3 == 0:
                pen = QPen(self.primary_color, 3)
                length = outer_radius * 0.15
            else:
                pen = QPen(self.secondary_color, 1)
                length = outer_radius * 0.08
                
            painter.setPen(pen)
            painter.drawLine(QPoint(int(outer_radius - length), 0), 
                           QPoint(int(outer_radius + 2), 0))
            painter.rotate(segment_angle)
        painter.restore()
        
        # Draw inner rotating elements
        painter.save()
        painter.translate(center)
        painter.rotate(self.inner_rotation)
        
        # Draw circular audio visualizer
        if self.is_listening or self.is_speaking:
            points_count = len(self.audio_data)
            angle_step = 360 / points_count
            
            for i, amplitude in enumerate(self.audio_data):
                angle = math.radians(i * angle_step)
                norm_amp = amplitude * 0.5  # Normalize amplitude
                
                # Calculate start and end points
                start_x = inner_radius * 0.7 * math.cos(angle)
                start_y = inner_radius * 0.7 * math.sin(angle)
                end_x = (inner_radius * 0.7 + norm_amp * inner_radius * 0.5) * math.cos(angle)
                end_y = (inner_radius * 0.7 + norm_amp * inner_radius * 0.5) * math.sin(angle)
                
                # Draw line
                if self.is_listening:
                    color = self.highlight_color
                else:
                    color = self.primary_color
                    
                painter.setPen(QPen(color, 1.5))
                painter.drawLine(QPoint(int(start_x), int(start_y)), 
                               QPoint(int(end_x), int(end_y)))
        
        painter.restore()
        
        # Draw status text
        painter.setPen(self.primary_color)
        font = QFont("Arial", 10)
        painter.setFont(font)
        
        status_text = "STANDBY"
        if self.is_listening:
            status_text = "LISTENING"
        elif self.is_processing:
            status_text = "PROCESSING"
        elif self.is_speaking:
            status_text = "SPEAKING"
            
        painter.drawText(
            QRectF(0, self.height() - 30, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter, 
            status_text
        )
        
        # Draw current time
        current_time = datetime.now().strftime("%H:%M:%S")
        painter.drawText(
            QRectF(0, 10, self.width(), 30),
            Qt.AlignmentFlag.AlignCenter, 
            current_time
        )


class WaveformVisualizer(FigureCanvas):
    """Waveform visualization for the audio input/output."""
    
    def __init__(self, parent=None, width=5, height=1, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        
        # Set up the style to match JARVIS
        fig.patch.set_facecolor('#001014')
        self.axes.set_facecolor('#001014')
        self.axes.spines['top'].set_visible(False)
        self.axes.spines['right'].set_visible(False)
        self.axes.spines['bottom'].set_visible(False)
        self.axes.spines['left'].set_visible(False)
        self.axes.tick_params(axis='both', colors='#00B4DC')
        
        # Initial empty plot
        self.x_data = np.linspace(0, 1, 100)
        self.y_data = np.zeros(100)
        self.line, = self.axes.plot(self.x_data, self.y_data, '-', lw=2, color='#00B4DC')
        
        self.axes.set_ylim(-1, 1)
        self.axes.set_xlim(0, 1)
        self.axes.get_xaxis().set_visible(False)
        self.axes.get_yaxis().set_visible(False)
        
        super().__init__(fig)
        self.setParent(parent)
        
    def update_waveform(self, data):
        """Update the waveform visualization with new audio data."""
        if len(data) > 0:
            # Normalize the data
            max_val = max(abs(max(data)), abs(min(data)))
            if max_val > 0:
                normalized_data = data / max_val
            else:
                normalized_data = data
                
            # Update the plot
            x_data = np.linspace(0, 1, len(normalized_data))
            self.line.set_xdata(x_data)
            self.line.set_ydata(normalized_data)
            self.axes.set_xlim(0, 1)
            self.draw()


class CommandLogWidget(QWidget):
    """Widget to display command history with JARVIS-style."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        
        # Main layout
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        
        # Title
        title_label = QLabel("COMMAND LOG")
        title_label.setStyleSheet("""
            color: #00B4DC;
            font-family: 'Arial';
            font-size: 12px;
            font-weight: bold;
        """)
        layout.addWidget(title_label)
        
        # Log area
        self.log_widget = QWidget()
        self.log_layout = QVBoxLayout(self.log_widget)
        self.log_layout.setContentsMargins(5, 5, 5, 5)
        self.log_layout.setSpacing(2)
        self.log_layout.addStretch()
        
        # Scroll area for the log
        log_frame = QFrame()
        log_frame.setObjectName("logFrame")
        log_frame.setStyleSheet("""
            #logFrame {
                background-color: #00101C;
                border: 1px solid #00B4DC;
                border-radius: 5px;
            }
        """)
        log_frame_layout = QVBoxLayout(log_frame)
        log_frame_layout.setContentsMargins(5, 5, 5, 5)
        log_frame_layout.addWidget(self.log_widget)
        
        layout.addWidget(log_frame)
        
    def add_user_message(self, message):
        """Add a user message to the command log."""
        time_str = datetime.now().strftime("%H:%M:%S")
        msg_widget = QWidget()
        msg_layout = QHBoxLayout(msg_widget)
        msg_layout.setContentsMargins(0, 0, 0, 0)
        
        # Time label
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #888888; font-size: 10px; min-width: 60px;")
        
        # User indicator
        user_label = QLabel("USER:")
        user_label.setStyleSheet("color: #00B4DC; font-weight: bold; min-width: 50px;")
        
        # Message content
        content_label = QLabel(message)
        content_label.setStyleSheet("color: #FFFFFF;")
        content_label.setWordWrap(True)
        
        msg_layout.addWidget(time_label)
        msg_layout.addWidget(user_label)
        msg_layout.addWidget(content_label, 1)
        
        self.log_layout.insertWidget(self.log_layout.count() - 1, msg_widget)
        
    def add_assistant_message(self, message):
        """Add an assistant message to the command log."""
        time_str = datetime.now().strftime("%H:%M:%S")
        msg_widget = QWidget()
        msg_layout = QHBoxLayout(msg_widget)
        msg_layout.setContentsMargins(0, 0, 0, 0)
        
        # Time label
        time_label = QLabel(time_str)
        time_label.setStyleSheet("color: #888888; font-size: 10px; min-width: 60px;")
        
        # Assistant indicator
        assistant_label = QLabel("DAISY:")
        assistant_label.setStyleSheet("color: #00E5FF; font-weight: bold; min-width: 50px;")
        
        # Message content
        content_label = QLabel(message)
        content_label.setStyleSheet("color: #CCFFFF;")
        content_label.setWordWrap(True)
        
        msg_layout.addWidget(time_label)
        msg_layout.addWidget(assistant_label)
        msg_layout.addWidget(content_label, 1)
        
        self.log_layout.insertWidget(self.log_layout.count() - 1, msg_widget)


class JarvisGUI(QMainWindow):
    """Main window for the JARVIS-inspired GUI."""
    
    # Signals for communication with assistant logic
    start_listening_signal = pyqtSignal()
    stop_listening_signal = pyqtSignal()
    
    def __init__(self):
        """Initialize the main window."""
        super().__init__()
        
        # Set up window
        self.setWindowTitle("D.A.I.S.Y. Voice Assistant")
        self.setMinimumSize(800, 600)
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #000810;
                color: #FFFFFF;
                font-family: 'Arial';
            }
            QPushButton {
                background-color: #00304D;
                color: #00B4DC;
                border: 1px solid #00B4DC;
                border-radius: 5px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #004066;
            }
            QPushButton:pressed {
                background-color: #005080;
            }
            QStatusBar {
                background-color: #001020;
                color: #00B4DC;
                border-top: 1px solid #003050;
            }
        """)
        
        # Add status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready", 3000)
        
        # Create central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Top section with title
        top_layout = QHBoxLayout()
        title_label = QLabel("D.A.I.S.Y. INTERFACE SYSTEM")
        title_label.setStyleSheet("""
            color: #00B4DC;
            font-size: 18px;
            font-weight: bold;
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        top_layout.addWidget(title_label)
        main_layout.addLayout(top_layout)
        
        # Main content layout
        content_layout = QHBoxLayout()
        
        # Left section with visualizer
        left_panel = QWidget()
        left_panel.setMinimumWidth(350)
        left_layout = QVBoxLayout(left_panel)
        
        # Audio visualizer
        self.audio_visualizer = AudioVisualizer()
        left_layout.addWidget(self.audio_visualizer)
        
        # Control buttons
        control_layout = QHBoxLayout()
        self.talk_button = QPushButton("Talk to DAISY")
        self.talk_button.clicked.connect(self.toggle_listening)
        control_layout.addWidget(self.talk_button)
        left_layout.addLayout(control_layout)
        
        # Status section
        status_widget = QWidget()
        status_layout = QVBoxLayout(status_widget)
        status_layout.setContentsMargins(0, 10, 0, 0)
        
        # Status title
        status_title = QLabel("SYSTEM STATUS")
        status_title.setStyleSheet("""
            color: #00B4DC;
            font-size: 12px;
            font-weight: bold;
        """)
        status_layout.addWidget(status_title)
        
        # Status indicators
        status_frame = QFrame()
        status_frame.setObjectName("statusFrame") 
        status_frame.setStyleSheet("""
            #statusFrame {
                background-color: #00101C;
                border: 1px solid #00B4DC;
                border-radius: 5px;
            }
        """)
        status_frame_layout = QVBoxLayout(status_frame)
        
        # System status indicators
        self.system_status = QLabel("System: Online")
        self.system_status.setStyleSheet("color: #00FF00;")
        
        self.speech_status = QLabel("Speech Recognition: Ready")
        self.speech_status.setStyleSheet("color: #00B4DC;")
        
        self.llm_status = QLabel("LLM Connection: Connected")
        self.llm_status.setStyleSheet("color: #00B4DC;")
        
        status_frame_layout.addWidget(self.system_status)
        status_frame_layout.addWidget(self.speech_status)
        status_frame_layout.addWidget(self.llm_status)
        
        status_layout.addWidget(status_frame)
        left_layout.addWidget(status_widget)
        
        # Add left panel to content layout
        content_layout.addWidget(left_panel)
        
        # Right section with command log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        
        # Command log
        self.command_log = CommandLogWidget()
        right_layout.addWidget(self.command_log)
        
        # Waveform visualizer
        waveform_title = QLabel("AUDIO WAVEFORM")
        waveform_title.setStyleSheet("""
            color: #00B4DC;
            font-size: 12px;
            font-weight: bold;
        """)
        right_layout.addWidget(waveform_title)
        
        self.waveform = WaveformVisualizer(width=5, height=1.5, dpi=100)
        right_layout.addWidget(self.waveform)
        
        # Add right panel to content layout
        content_layout.addWidget(right_panel)
        
        # Add content layout to main layout
        main_layout.addLayout(content_layout)
        
        # Bottom section with current transcription
        self.transcription_label = QLabel("")
        self.transcription_label.setStyleSheet("""
            background-color: #00101C;
            border: 1px solid #00B4DC;
            border-radius: 5px;
            padding: 8px;
            color: #FFFFFF;
        """)
        self.transcription_label.setMinimumHeight(50)
        self.transcription_label.setWordWrap(True)
        main_layout.addWidget(self.transcription_label)
        
        # Sample data for demo
        self.initialize_demo_data()
        
    def initialize_demo_data(self):
        """Initialize with some demo data for visualization."""
        # Demo audio data
        sample_data = np.sin(np.linspace(0, 4*np.pi, 100)) * 0.5
        noise = np.random.normal(0, 0.1, 100)
        audio_data = sample_data + noise
        self.waveform.update_waveform(audio_data)
        
        # Demo frequency data for the circular visualizer
        freq_data = np.abs(np.fft.rfft(audio_data))
        normalized_freq = freq_data / (np.max(freq_data) + 1e-10)
        self.audio_visualizer.update_audio_data(normalized_freq)
        
        # Demo command log
        self.command_log.add_user_message("Hey DAISY, what's the weather like today?")
        self.command_log.add_assistant_message("Current temperature is 72°F with partly cloudy skies.")
        
    def toggle_listening(self):
        """Toggle the listening state."""
        if self.audio_visualizer.is_listening:
            # Stop listening
            self.audio_visualizer.set_state(listening=False)
            self.talk_button.setText("Talk to DAISY")
            self.speech_status.setText("Speech Recognition: Ready")
            self.transcription_label.setText("")
            self.stop_listening_signal.emit()
        else:
            # Start listening
            self.audio_visualizer.set_state(listening=True)
            self.talk_button.setText("Stop Listening")
            self.speech_status.setText("Speech Recognition: Listening")
            self.transcription_label.setText("Listening...")
            self.start_listening_signal.emit()
    
    def set_processing_state(self):
        """Set the UI to processing state."""
        self.audio_visualizer.set_state(processing=True)
        self.speech_status.setText("Speech Recognition: Processing")
    
    def set_speaking_state(self):
        """Set the UI to speaking state."""
        self.audio_visualizer.set_state(speaking=True)
        self.speech_status.setText("Speech Recognition: Assistant Speaking")
    
    def update_transcription(self, text):
        """Update the transcription text in the GUI."""
        self.transcription_label.setText(text)
        
    def update_status_message(self, message):
        """Display a status message in the GUI."""
        if hasattr(self, 'status_bar'):
            self.status_bar.showMessage(message, 5000)  # Show for 5 seconds
        print(f"Status: {message}")  # Also print to console
        
    def add_user_input(self, text):
        """Add user message to conversation history."""
        self.command_log.add_user_message(text)
        
    def add_assistant_response(self, text):
        """Add assistant response to the command log."""
        self.command_log.add_assistant_message(text)
        
    def update_audio_visualization(self, audio_data):
        """Update audio visualizations with new data."""
        if len(audio_data) > 0:
            # Update waveform
            self.waveform.update_waveform(audio_data)
            
            # Update frequency visualization
            freq_data = np.abs(np.fft.rfft(audio_data))
            normalized_freq = freq_data / (np.max(freq_data) + 1e-10)
            self.audio_visualizer.update_audio_data(normalized_freq) 