from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QPushButton, QGroupBox, QRadioButton,
                             QListWidget, QMessageBox, QTabWidget, QSpinBox, QFormLayout,
                             QTextEdit, QFileDialog, QGridLayout, QApplication, QSplitter,
                             QFrame, QScrollArea, QComboBox, QListWidgetItem, QStyleFactory, QSizePolicy,
                             QTextBrowser, QProgressBar, QTableWidget, QTableWidgetItem, QHeaderView,
                             QDialog, QPlainTextEdit, QDialogButtonBox)
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal, QSize, QMargins, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIntValidator, QColor, QPalette, QLinearGradient, QIcon, QPixmap, QFontDatabase
import os
import sqlite3
import random
import time
from genetic_algorithm import GeneticOptimizer

from simulated_annealing import SimulatedAnnealingOptimizer
from greedy_optimizer import GreedyOptimizer
from solution_validator import SolutionValidator

# 添加用于生成PDF的库
try:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import letter
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch

    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False


class ComputationThread(QThread):
    # Define signals for thread communication
    result_ready = pyqtSignal(list, float)  # Signal for results and execution time
    progress_update = pyqtSignal(str)  # Signal for progress updates
    error_occurred = pyqtSignal(str)  # Signal for error handling
    progress_value = pyqtSignal(int)  # Signal for progress bar value (0-100)

    def __init__(self, samples, j, s, k, f, algorithm):
        super().__init__()
        self.samples = samples
        self.j = j
        self.s = s
        self.k = k
        self.f = f
        self.algorithm = algorithm
        self.stopped = False

    def stop(self):
        """Set the stopped flag to pause the thread"""
        self.stopped = True
        self.progress_update.emit("Calculation paused by user")

    def run(self):
        try:
            # Emit progress update
            self.progress_update.emit("Calculation started, please wait...\n")
            self.progress_value.emit(0)  # Start progress at 0%

            # Record start time
            start_time = time.time()

            # Create optimizer based on selected algorithm
            if self.algorithm == "genetic_algorithm":
                optimizer = GeneticOptimizer(self.samples, self.j, self.s, self.k, self.f)
            elif self.algorithm == "simulated_annealing":
                optimizer = SimulatedAnnealingOptimizer(self.samples, self.j, self.s, self.k, self.f)
            else:  # greedy_algorithm
                optimizer = GreedyOptimizer(self.samples, self.j, self.s, self.k, self.f)
            # Setup progress callback for the optimizer
            optimizer.set_progress_callback(self.update_progress)

            # Run optimization
            best_solution = optimizer.optimize()

            # Check if stopped
            if self.stopped:
                self.progress_update.emit("Calculation was paused. Results may be incomplete.")

            # Calculate execution time
            execution_time = time.time() - start_time

            # Emit final progress
            self.progress_value.emit(100)  # Complete the progress

            # Emit result with execution time
            self.result_ready.emit(best_solution, execution_time)

        except Exception as e:
            # Emit error signal if exception occurs
            self.error_occurred.emit(str(e))

    def update_progress(self, progress_percent, status_message=None):
        """Called by optimizer to update progress"""
        self.progress_value.emit(progress_percent)
        if status_message:
            self.progress_update.emit(status_message)

        # Check if thread should stop
        return not self.stopped


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("An Optimal Samples Selection System")
        self.setMinimumSize(1000, 1200)
        self.settings = QSettings("OptimalSamples", "SampleSelection")

        # Set application style
        self.set_application_style()

        # Database initialization
        self.db_path = "results.db"
        self.initialize_database()

        # Initialize attributes
        self.samples = []
        self.current_results = []
        self.current_run_id = 0
        self.execution_time = 0.0

        # Create main layout
        self.setup_ui()

        # Load saved results
        self.load_saved_runs()

    def set_application_style(self):
        """Set application style"""
        # Try to set application font
        app_font = QFont("Segoe UI", 10)
        QApplication.setFont(app_font)

        # Set global stylesheet with simplified design
        self.setStyleSheet("""
            QMainWindow, QWidget {
                background-color: #f5f5f5;
                color: #333333;
                font-size: 10pt;
            }

            QTabWidget::pane {
                border: 1px solid #cccccc;
                border-radius: 4px;
                background-color: #ffffff;
                padding: 5px;
            }

            QTabBar::tab {
                background-color: #e0e0e0;
                color: #333333;
                border: 1px solid #cccccc;
                padding: 8px 16px;
                margin: 1px;
                border-top-left-radius: 4px;
                border-top-right-radius: 4px;
            }

            QTabBar::tab:selected {
                background-color: #ffffff;
                border-bottom-color: #ffffff;
            }

            QGroupBox {
                border: 1px solid #cccccc;
                border-radius: 4px;
                margin-top: 16px;
                padding: 10px;
                background-color: #ffffff;
            }

            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 5px;
                background-color: #ffffff;
            }

            QPushButton {
                background-color: #2196f3;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 12px;
                font-weight: bold;
            }

            QPushButton:hover {
                background-color: #0d8aee;
            }

            QPushButton:pressed {
                background-color: #0c7cd5;
            }

            QPushButton:disabled {
                background-color: #cccccc;
                color: #888888;
            }

            QLineEdit, QSpinBox, QTextEdit {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 4px;
                padding: 5px;
            }

            QTextEdit, QTextBrowser {
                background-color: #ffffff;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }

            QListWidget {
                background-color: #ffffff;
                color: #333333;
                border: 1px solid #cccccc;
                border-radius: 4px;
                alternate-background-color: #f0f0f0;
            }

            QListWidget::item {
                height: 28px;
                padding: 4px;
                border-bottom: 1px solid #eeeeee;
            }

            QListWidget::item:selected {
                background-color: #2196f3;
                color: white;
            }

            QLabel {
                color: #333333;
            }

            QRadioButton {
                color: #333333;
                spacing: 8px;
            }

            QRadioButton::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 1px solid #cccccc;
            }

            QRadioButton::indicator:checked {
                background-color: #2196f3;
                border: 4px solid #ffffff;
            }

            QScrollBar:vertical {
                border: none;
                background-color: #f0f0f0;
                width: 10px;
            }

            QScrollBar::handle:vertical {
                background-color: #cccccc;
                border-radius: 5px;
                min-height: 20px;
            }

            QScrollBar::handle:vertical:hover {
                background-color: #bbbbbb;
            }
        """)

    def setup_ui(self):
        """Set user interface"""
        # Create central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(15)

        # Title area - 简化设计
        title_frame = QFrame()
        title_layout = QVBoxLayout(title_frame)
        title_layout.setContentsMargins(0, 0, 0, 10)
        title_layout.setSpacing(5)

        # Title label
        title_label = QLabel("An Optimal Samples Selection System")
        title_font = QFont("Segoe UI", 18)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #333333;")

        # Subtitle - 更简洁的副标题
        subtitle_label = QLabel("Multiple Algorithm Optimization")
        subtitle_label.setAlignment(Qt.AlignCenter)
        subtitle_label.setStyleSheet("color: #666666; font-size: 12pt;")

        # 添加简单的分隔线
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setFrameShadow(QFrame.Sunken)
        separator.setStyleSheet("background-color: #dddddd; max-height: 1px;")

        title_layout.addWidget(title_label)
        title_layout.addWidget(subtitle_label)
        title_layout.addWidget(separator)
        main_layout.addWidget(title_frame)

        # Create tab widget
        tab_widget = QTabWidget()
        tab_widget.setDocumentMode(True)
        tab_widget.setTabPosition(QTabWidget.North)
        main_layout.addWidget(tab_widget, 1)

        # First tab - Parameter Setting and Calculation
        input_tab = QWidget()
        tab_widget.addTab(input_tab, "Parameter Setting")

        # Second tab - Result Management
        results_tab = QWidget()
        tab_widget.addTab(results_tab, "Result Management")

        # Third tab - Algorithm Comparison
        comparison_tab = QWidget()
        tab_widget.addTab(comparison_tab, "Algorithm Comparison")

        # Set parameter input page
        self.setup_input_tab(input_tab)

        # Set result management page
        self.setup_results_tab(results_tab)

        # Set algorithm comparison page
        self.setup_comparison_tab(comparison_tab)

        # Add status bar
        status_bar = self.statusBar()
        status_bar.showMessage("System Ready", 3000)
        status_bar.setStyleSheet("""
            QStatusBar {
                background-color: #f5f5f5;
                color: #666666;
                border-top: 1px solid #dddddd;
            }
        """)

    def setup_input_tab(self, tab):
        """Set parameter input tab page"""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setFrameShape(QFrame.NoFrame)

        # Create main container
        content_widget = QWidget()
        scroll_area.setWidget(content_widget)

        # Main layout
        main_layout = QVBoxLayout(tab)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(scroll_area)

        # Content layout
        layout = QVBoxLayout(content_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # Parameter setting area
        param_group = QGroupBox("Algorithm Parameter Setting")

        # Use grid layout, three parameters per row
        param_layout = QGridLayout()
        param_layout.setSpacing(15)
        param_layout.setContentsMargins(20, 30, 20, 20)

        # Create label style
        label_style = """
            QLabel {
                font-size: 11pt;
                color: #333333;
                font-weight: normal;
            }
        """

        # Create input fields - First row
        # m parameter
        m_label = QLabel("m (45<=m<=54):")
        m_label.setStyleSheet(label_style)
        m_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.m_input = QSpinBox()
        self.m_input.setRange(45, 54)
        self.m_input.setValue(45)
        param_layout.addWidget(m_label, 0, 0)
        param_layout.addWidget(self.m_input, 0, 1)

        # n parameter
        n_label = QLabel("n (7<=n<=25):")
        n_label.setStyleSheet(label_style)
        n_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.n_input = QSpinBox()
        self.n_input.setRange(7, 25)
        self.n_input.setValue(7)
        param_layout.addWidget(n_label, 0, 2)
        param_layout.addWidget(self.n_input, 0, 3)

        # k parameter
        k_label = QLabel("k (4<=k<=7):")
        k_label.setStyleSheet(label_style)
        k_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.k_input = QSpinBox()
        self.k_input.setRange(4, 7)
        self.k_input.setValue(6)  # Default value is 6
        param_layout.addWidget(k_label, 0, 4)
        param_layout.addWidget(self.k_input, 0, 5)

        # Second row
        # j parameter
        j_label = QLabel("j (s<=j<=k):")
        j_label.setStyleSheet(label_style)
        j_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.j_input = QSpinBox()
        self.j_input.setRange(3, 25)
        self.j_input.setValue(4)
        param_layout.addWidget(j_label, 1, 0)
        param_layout.addWidget(self.j_input, 1, 1)

        # s parameter
        s_label = QLabel("s (3<=s<=7):")
        s_label.setStyleSheet(label_style)
        s_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.s_input = QSpinBox()
        self.s_input.setRange(3, 7)
        self.s_input.setValue(4)
        param_layout.addWidget(s_label, 1, 2)
        param_layout.addWidget(self.s_input, 1, 3)

        # f parameter
        f_label = QLabel("at least f s samples:")
        f_label.setStyleSheet(label_style)
        f_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.f_input = QSpinBox()
        self.f_input.setRange(1, 10)
        self.f_input.setValue(1)
        param_layout.addWidget(f_label, 1, 4)
        param_layout.addWidget(self.f_input, 1, 5)

        # Third row - Algorithm selection
        algorithm_label = QLabel("Algorithm:")
        algorithm_label.setStyleSheet(label_style)
        algorithm_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.algorithm_combo = QComboBox()
        self.algorithm_combo.addItem("Genetic Algorithm")
        self.algorithm_combo.addItem("Simulated Annealing")
        self.algorithm_combo.addItem("Greedy Algorithm")
        self.algorithm_combo.currentIndexChanged.connect(self.on_algorithm_changed)
        param_layout.addWidget(algorithm_label, 2, 0)
        param_layout.addWidget(self.algorithm_combo, 2, 1)

        # Set parameter constraints
        self.m_input.valueChanged.connect(self.update_n_max)
        self.n_input.valueChanged.connect(self.update_constraints)
        self.j_input.valueChanged.connect(self.update_constraints)
        self.s_input.valueChanged.connect(self.update_constraints)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Top selection and button area
        top_controls = QWidget()
        top_layout = QHBoxLayout(top_controls)

        # Selection method (left side)
        selection_frame = QFrame()
        selection_layout = QHBoxLayout(selection_frame)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        self.random_select = QRadioButton("Random N")
        self.random_select.setChecked(True)

        self.manual_select = QRadioButton("Input N")

        selection_layout.addWidget(self.random_select)
        selection_layout.addWidget(self.manual_select)
        selection_layout.addStretch()

        top_layout.addWidget(selection_frame)
        top_layout.addStretch()

        # Buttons (right side)
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)

        self.calculate_btn = QPushButton("Execute")
        self.calculate_btn.clicked.connect(self.calculate_optimal_groups)
        buttons_layout.addWidget(self.calculate_btn)

        self.stop_btn = QPushButton("Stop")
        self.stop_btn.clicked.connect(self.stop_calculation)
        self.stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.stop_btn)

        self.save_btn = QPushButton("Store")
        self.save_btn.clicked.connect(self.save_results)
        self.save_btn.setEnabled(False)
        buttons_layout.addWidget(self.save_btn)

        self.clear_btn = QPushButton("Clear")
        self.clear_btn.clicked.connect(self.clear_input_and_results)
        buttons_layout.addWidget(self.clear_btn)

        top_layout.addWidget(buttons_frame)

        layout.addWidget(top_controls)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(False)  # Hide progress bar initially
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #2196f3;
                width: 10px;
                margin: 0.5px;
            }
        """)
        layout.addWidget(self.progress_bar)

        # User input area for manual entry
        user_input_group = QGroupBox("User Input")
        user_input_layout = QVBoxLayout(user_input_group)

        # Manual input area with custom styling
        manual_input_layout = QHBoxLayout()
        self.sample_input = QLineEdit()
        self.sample_input.setPlaceholderText("Enter sample numbers separated by commas (e.g., 01,02,03)")
        self.sample_input.setEnabled(False)
        manual_input_layout.addWidget(self.sample_input)

        user_input_layout.addLayout(manual_input_layout)

        # Connect signals
        self.random_select.toggled.connect(self.toggle_input_mode)
        self.manual_select.toggled.connect(self.toggle_input_mode)
        self.n_input.valueChanged.connect(self.update_n_dependent_ui)
        self.n_input.valueChanged.connect(self.populate_default_samples)

        layout.addWidget(user_input_group)

        # Main content area with samples and results side by side
        content_area = QWidget()
        content_layout = QHBoxLayout(content_area)
        content_layout.setContentsMargins(0, 0, 0, 0)

        # Left side - Values Input
        values_group = QGroupBox("Values Input")
        values_layout = QVBoxLayout(values_group)

        self.samples_display = QTextEdit()
        self.samples_display.setReadOnly(True)
        self.samples_display.setMinimumHeight(200)

        values_layout.addWidget(self.samples_display)

        # Right side - Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.result_display = QTextBrowser()
        self.result_display.setReadOnly(True)
        self.result_display.setMinimumHeight(200)

        results_layout.addWidget(self.result_display)

        # Add both sides to content layout
        content_layout.addWidget(values_group)
        content_layout.addWidget(results_group)

        layout.addWidget(content_area)

        # Results ID display
        id_layout = QHBoxLayout()
        id_layout.setContentsMargins(0, 0, 0, 0)

        self.id_label = QLabel("")
        id_layout.addWidget(self.id_label, 0, Qt.AlignCenter)

        layout.addLayout(id_layout)

    def populate_default_samples(self, n_value):
        """Generate a default sequential list of samples based on n value."""
        if n_value <= 0:
            self.sample_input.setText("")
            return

        # Generate samples with leading zeros for single digits (01, 02, etc.)
        samples = [f"{i:02d}" for i in range(1, n_value + 1)]
        sample_text = ", ".join(samples)
        self.sample_input.setText(sample_text)

    def setup_results_tab(self, tab):
        """Set result management tab page"""
        # Create main layout
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # Create horizontal splitter
        splitter = QSplitter(Qt.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        # Saved run record panel
        runs_panel = QWidget()
        runs_layout = QVBoxLayout(runs_panel)
        runs_layout.setContentsMargins(0, 0, 10, 0)

        runs_group = QGroupBox("Saved Run Records")

        runs_inner_layout = QVBoxLayout(runs_group)
        runs_inner_layout.setContentsMargins(15, 30, 15, 15)
        runs_inner_layout.setSpacing(10)

        # List header
        list_header = QLabel("Select a Record to View Details")
        list_header.setAlignment(Qt.AlignCenter)
        runs_inner_layout.addWidget(list_header)

        # Run record list
        self.runs_list = QListWidget()
        self.runs_list.setAlternatingRowColors(True)
        self.runs_list.itemSelectionChanged.connect(self.load_selected_run)

        # 禁用垂直和水平滚动条
        self.runs_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.runs_list.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        # 设置自动包装模式，确保文本不会超出列表项宽度
        self.runs_list.setTextElideMode(Qt.ElideNone)
        self.runs_list.setWordWrap(True)

        # 设置最小宽度，确保有足够空间显示长ID
        self.runs_list.setMinimumWidth(250)

        # 设置最小高度
        self.runs_list.setMinimumHeight(300)

        runs_inner_layout.addWidget(self.runs_list)

        runs_layout.addWidget(runs_group)
        splitter.addWidget(runs_panel)

        # Result details panel
        details_panel = QWidget()
        details_layout = QVBoxLayout(details_panel)
        details_layout.setContentsMargins(10, 0, 0, 0)

        details_group = QGroupBox("Result Details")

        details_inner_layout = QVBoxLayout(details_group)
        details_inner_layout.setContentsMargins(15, 30, 15, 15)
        details_inner_layout.setSpacing(15)

        # Details text display area
        self.details_display = QTextBrowser()
        self.details_display.setReadOnly(True)
        self.details_display.setMinimumHeight(350)
        self.details_display.setTextInteractionFlags(Qt.TextBrowserInteraction)
        self.details_display.setAcceptRichText(True)
        details_inner_layout.addWidget(self.details_display)

        # Button area
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 5, 0, 0)

        buttons_layout.addStretch()

        # Delete button
        self.delete_btn = QPushButton("Delete Selected Record")
        self.delete_btn.setMinimumSize(150, 40)
        self.delete_btn.clicked.connect(self.delete_selected_run)
        self.delete_btn.setEnabled(False)
        buttons_layout.addWidget(self.delete_btn)

        # Export button
        self.export_btn = QPushButton("Export as Text File")
        self.export_btn.setMinimumSize(150, 40)
        self.export_btn.clicked.connect(self.export_results)
        self.export_btn.setEnabled(False)
        buttons_layout.addWidget(self.export_btn)

        details_inner_layout.addWidget(buttons_frame)
        details_layout.addWidget(details_group)

        splitter.addWidget(details_panel)

        # Set split ratio - Give details panel more space
        splitter.setSizes([350, 550])

        layout.addWidget(splitter)

    def toggle_input_mode(self):
        """Toggle between random and manual input modes."""
        if self.manual_select.isChecked():
            self.sample_input.setEnabled(True)
            # Populate default samples when switching to manual mode
            self.populate_default_samples(self.n_input.value())
        else:
            self.sample_input.setEnabled(False)
            self.sample_input.setText("")

    def update_n_max(self):
        """Update the maximum allowed value for n input based on m."""
        m_value = self.m_input.value()
        # n can't be larger than m
        self.n_input.setMaximum(m_value)
        # If current n is larger than m, adjust it
        if self.n_input.value() > m_value:
            self.n_input.setValue(m_value)

        # Update UI elements that depend on n
        n_value = self.n_input.value()
        self.update_n_dependent_ui(n_value)

        # Update default samples if manual selection is enabled
        if hasattr(self, 'manual_select') and self.manual_select.isChecked():
            self.populate_default_samples(n_value)

    def update_constraints(self):
        """Update j, s, k constraints"""
        n_value = self.n_input.value()
        j_value = self.j_input.value()
        s_value = self.s_input.value()

        # Ensure j >= s
        if j_value < s_value:
            self.j_input.setValue(s_value)

        # Ensure j <= n, k <= n
        self.j_input.setMaximum(n_value)
        self.k_input.setMaximum(min(7, n_value))

        # Ensure s <= j, s <= k
        s_max = min(self.j_input.value(), self.k_input.value())
        self.s_input.setMaximum(min(7, s_max))

    def initialize_database(self):
        """Initialize database"""
        conn = sqlite3.connect(self.db_path)

        # Enable foreign key constraints
        conn.execute("PRAGMA foreign_keys = ON")

        cursor = conn.cursor()

        # Create runs table with run_count and formatted_id fields
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS runs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            m INTEGER,
            n INTEGER,
            k INTEGER,
            j INTEGER,
            s INTEGER,
            f INTEGER,
            run_id INTEGER,
            timestamp TEXT,
            sample_count INTEGER,
            execution_time REAL,
            algorithm TEXT,
            run_count INTEGER DEFAULT 1,
            formatted_id TEXT
        )
        ''')

        # Create samples table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS samples (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            sample TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE
        )
        ''')

        # Create results table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER,
            group_id INTEGER,
            sample TEXT,
            FOREIGN KEY (run_id) REFERENCES runs (id) ON DELETE CASCADE
        )
        ''')

        # Check if algorithm column exists in runs table
        cursor.execute("PRAGMA table_info(runs)")
        columns = [info[1] for info in cursor.fetchall()]

        # Add algorithm column if it doesn't exist
        if 'algorithm' not in columns:
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN algorithm TEXT DEFAULT 'genetic'")
            except sqlite3.Error as e:
                print(f"Error adding algorithm column: {e}")

        # Add run_count column if it doesn't exist
        if 'run_count' not in columns:
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN run_count INTEGER DEFAULT 1")
            except sqlite3.Error as e:
                print(f"Error adding run_count column: {e}")

        # Add formatted_id column if it doesn't exist
        if 'formatted_id' not in columns:
            try:
                cursor.execute("ALTER TABLE runs ADD COLUMN formatted_id TEXT")
            except sqlite3.Error as e:
                print(f"Error adding formatted_id column: {e}")

        conn.commit()
        conn.close()

    def load_saved_runs(self):
        """Load saved run records"""
        self.runs_list.clear()

        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()

            # Check if run_count and formatted_id columns exist
            cursor.execute("PRAGMA table_info(runs)")
            columns = [info[1] for info in cursor.fetchall()]

            # Select query based on available columns
            if 'run_count' in columns and 'formatted_id' in columns:
                query = '''
                SELECT id, m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time, run_count, formatted_id 
                FROM runs 
                ORDER BY timestamp DESC
                '''
            else:
                query = '''
                SELECT id, m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time
                FROM runs 
                ORDER BY timestamp DESC
                '''

            cursor.execute(query)
            rows = cursor.fetchall()

            # 设置QListWidget的垂直滚动条策略
            if len(rows) <= 10:
                # 如果记录少于10条，禁用垂直滚动条
                self.runs_list.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
            else:
                # 如果记录多于10条，启用垂直滚动条
                self.runs_list.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

            has_new_columns = 'run_count' in columns and 'formatted_id' in columns

            for row in rows:
                run_id = row[0]
                m, n, k, j, s, f = row[1:7]
                algorithm_run_id, timestamp, sample_count, execution_time = row[7:11]

                # Get run_count and formatted_id if available
                if has_new_columns:
                    run_count = row[11]
                    formatted_id = row[12]
                else:
                    run_count = 1
                    formatted_id = None

                # Check if results exist for old records
                if not has_new_columns:
                    cursor.execute('SELECT COUNT(*) FROM results WHERE run_id = ?', (run_id,))
                    sample_count = cursor.fetchone()[0]

                # Create list item
                item = QListWidgetItem(self.runs_list)
                item.setData(Qt.UserRole, run_id)

                # Use the formatted ID if available, otherwise generate one
                if formatted_id:
                    run_format = formatted_id
                else:
                    # For backward compatibility with old records
                    run_format = f"{m}-{n}-{k}-{j}-{s}-{run_count}-{sample_count}"

                # Set the text directly to the item and center it
                item.setText(run_format)
                item.setTextAlignment(Qt.AlignCenter)

                # Store additional information as user data for details view
                item.setData(Qt.UserRole + 1, timestamp)
                item.setData(Qt.UserRole + 2, execution_time)
                item.setData(Qt.UserRole + 3, f)
                item.setData(Qt.UserRole + 4, run_count if has_new_columns else 1)

                # Ensure items are sized properly
                item.setSizeHint(QSize(self.runs_list.width() - 30, 40))

            conn.close()

            # Set alternating row colors
            for i in range(self.runs_list.count()):
                if i % 2 == 0:
                    self.runs_list.item(i).setBackground(QColor("#f9f9f9"))
                else:
                    self.runs_list.item(i).setBackground(QColor("#f0f0f0"))

        except Exception as e:
            QMessageBox.warning(self, "Loading Error", f"Error loading run records: {str(e)}")

    def calculate_optimal_groups(self):
        """Calculate optimal combinations"""
        # Get parameter values
        m = self.m_input.value()
        n = self.n_input.value()
        k = self.k_input.value()
        j = self.j_input.value()
        s = self.s_input.value()
        f = self.f_input.value()
        algorithm = self.algorithm_combo.currentText().lower().replace(" ", "_")

        # Validate parameters
        if not (3 <= s <= j <= k <= n <= m):
            QMessageBox.warning(self, "Parameter Error",
                                "Parameters must satisfy: 3 ≤ s ≤ j ≤ k ≤ n ≤ m")
            return

        # Prepare samples
        self.samples = []
        if self.random_select.isChecked():
            # Randomly select samples
            all_samples = [f"{i:02d}" for i in range(1, m + 1)]
            self.samples = random.sample(all_samples, n)
        else:
            # Manual input samples
            input_text = self.sample_input.text().strip()
            if not input_text:
                QMessageBox.warning(self, "Input Error", "Please enter sample numbers")
                return

            try:
                input_samples = input_text.split(',')
                self.samples = [s.strip().zfill(2) for s in input_samples]

                # Validate samples
                if len(self.samples) != n:
                    QMessageBox.warning(self, "Input Error", f"Please enter exactly {n} samples")
                    return

                if len(set(self.samples)) != len(self.samples):
                    QMessageBox.warning(self, "Input Error", "Sample numbers cannot be duplicated")
                    return

                if any(not s.isdigit() or int(s) < 1 or int(s) > m for s in self.samples):
                    QMessageBox.warning(self, "Input Error", f"Sample numbers must be between 01-{m:02d}")
                    return
            except:
                QMessageBox.warning(self, "Input Error", "Please enter valid sample numbers, separated by commas")
                return

        # Display selected samples in numbered format
        self.samples.sort()
        samples_with_numbers = []
        for i, sample in enumerate(self.samples, 1):
            samples_with_numbers.append(f"{i}# {sample}")

        self.samples_display.setText("\n".join(samples_with_numbers))

        # Disable calculate button and enable stop button
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Processing...")
        self.stop_btn.setEnabled(True)

        # Clear result display
        self.result_display.clear()

        # Display calculation parameters
        self.result_display.append("Calculation started, please wait...\n")

        # Create current run ID
        self.current_run_id = int(time.time())

        # Check for run count from database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Find max run count for these parameters
            cursor.execute('''
            SELECT MAX(run_count) FROM runs 
            WHERE m = ? AND n = ? AND k = ? AND j = ? AND s = ? AND algorithm = ?
            ''', (m, n, k, j, s, algorithm))

            result = cursor.fetchone()
            run_count = 1  # Default is 1st run

            if result[0] is not None:
                run_count = result[0] + 1  # Next run

            conn.close()

            # Format and display run ID
            run_format = f"{m}-{n}-{k}-{j}-{s}-{run_count}-?"
            self.id_label.setText(run_format)

        except Exception:
            # If database query fails, just use a placeholder
            run_format = f"{m}-{n}-{k}-{j}-{s}-?-?"
            self.id_label.setText(run_format)

        # Show the progress bar
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        try:
            # Create and configure calculation thread
            self.computation_thread = ComputationThread(self.samples, j, s, k, f, algorithm)

            # Connect signals and slots
            self.computation_thread.result_ready.connect(self.handle_result)
            self.computation_thread.progress_update.connect(self.update_progress)
            self.computation_thread.progress_value.connect(self.progress_bar.setValue)
            self.computation_thread.error_occurred.connect(self.handle_error)

            # Start thread
            self.computation_thread.start()
        except Exception as e:
            self.result_display.append(f"Calculation error: {str(e)}")
            self.calculate_btn.setEnabled(True)
            self.calculate_btn.setText("Execute")
            self.progress_bar.setVisible(False)

    def stop_calculation(self):
        """Stop the running calculation thread"""
        if hasattr(self, 'computation_thread') and self.computation_thread.isRunning():
            self.computation_thread.stop()
            self.stop_btn.setEnabled(False)
            self.calculate_btn.setEnabled(True)
            self.calculate_btn.setText("Execute")
            self.result_display.append("Calculation paused by user. Results may be incomplete.")

    def handle_result(self, results, execution_time):
        """Handle calculation result"""
        self.current_results = results
        self.execution_time = execution_time  # Save execution time

        # Update run ID with result count
        m = self.m_input.value()
        n = self.n_input.value()
        k = self.k_input.value()
        j = self.j_input.value()
        s = self.s_input.value()
        run_format = f"{m}-{n}-{k}-{j}-{s}-1-{len(results)}"
        self.id_label.setText(run_format)

        # Display result and execution time
        self.result_display.clear()

        # Create simple numbered list of results
        result_text = []
        for i, group in enumerate(results, 1):
            result_text.append(f"{i}# {', '.join(group)}")

        if not results:
            result_text.append("No combinations found.")

        # Simple text display
        self.result_display.setText("\n".join(result_text))

        # Enable store button
        self.save_btn.setEnabled(True)

        # Restore calculate button and disable stop button
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Execute")
        self.stop_btn.setEnabled(False)

        # Hide progress bar
        self.progress_bar.setVisible(False)

    def update_progress(self, message):
        """Update progress information"""
        self.result_display.append(message)

    def handle_error(self, error_message):
        """Handle calculation error"""
        self.result_display.append(f"Calculation error: {error_message}")
        # Reset buttons
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Execute")
        self.stop_btn.setEnabled(False)

    def save_results(self):
        """Save calculation results to database"""
        if not self.current_results:
            return

        # Get parameters
        m = self.m_input.value()
        n = self.n_input.value()
        k = self.k_input.value()
        j = self.j_input.value()
        s = self.s_input.value()
        f = self.f_input.value()
        timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
        sample_count = len(self.current_results)
        algorithm = self.algorithm_combo.currentText()

        try:
            conn = sqlite3.connect(self.db_path)

            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")

            cursor = conn.cursor()

            # Default run count (in case query fails)
            run_count = 1

            try:
                # Check if there are runs with the same parameters
                cursor.execute('''
                SELECT MAX(run_count) FROM runs 
                WHERE m = ? AND n = ? AND k = ? AND j = ? AND s = ? AND algorithm = ?
                ''', (m, n, k, j, s, algorithm))

                result = cursor.fetchone()
                if result[0] is not None:
                    run_count = result[0] + 1  # Increment the run count
            except sqlite3.Error:
                # If column doesn't exist or any other error, use default run count
                run_count = 1

            # Generate the formatted run ID
            run_format = f"{m}-{n}-{k}-{j}-{s}-{run_count}-{sample_count}"

            try:
                # Try to insert with run_count and formatted_id
                cursor.execute('''
                INSERT INTO runs (m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time, run_count, formatted_id, algorithm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (m, n, k, j, s, f, self.current_run_id, timestamp, sample_count, self.execution_time, run_count,
                      run_format, algorithm))
            except sqlite3.Error:
                # Fallback to old schema if new columns don't exist
                cursor.execute('''
                INSERT INTO runs (m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time, algorithm)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (m, n, k, j, s, f, self.current_run_id, timestamp, sample_count, self.execution_time, algorithm))

            run_id = cursor.lastrowid

            # Insert samples
            for sample in self.samples:
                cursor.execute('''
                INSERT INTO samples (run_id, sample)
                VALUES (?, ?)
                ''', (run_id, sample))

            # Insert results
            for group_id, group in enumerate(self.current_results, 1):
                for sample in group:
                    cursor.execute('''
                    INSERT INTO results (run_id, group_id, sample)
                    VALUES (?, ?, ?)
                    ''', (run_id, group_id, sample))

            conn.commit()

            # Update the ID label with the saved formatted ID
            self.id_label.setText(run_format)

            # Update run record list
            self.load_saved_runs()

            # Disable save button
            self.save_btn.setEnabled(False)

            QMessageBox.information(self, "Save Successful", "Results have been successfully saved to the database")
        except Exception as e:
            QMessageBox.warning(self, "Save Failed", f"Error saving results: {str(e)}")
        finally:
            if conn:
                conn.close()

    def load_selected_run(self):
        """Load selected run record"""
        selected_items = self.runs_list.selectedItems()
        if not selected_items:
            self.details_display.clear()
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)
            return

        # Get selected run ID
        run_id = selected_items[0].data(Qt.UserRole)

        # Enable buttons
        self.delete_btn.setEnabled(True)
        self.export_btn.setEnabled(True)

        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()

            # Check if run_count and formatted_id columns exist
            cursor.execute("PRAGMA table_info(runs)")
            columns = [info[1] for info in cursor.fetchall()]
            has_new_columns = 'run_count' in columns and 'formatted_id' in columns

            # Get run details including formatted_id if available
            if has_new_columns:
                query = '''
                SELECT m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time, run_count, formatted_id, algorithm
                FROM runs
                WHERE id = ?
                '''
            else:
                query = '''
                SELECT m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time, algorithm
                FROM runs
                WHERE id = ?
                '''

            cursor.execute(query, (run_id,))

            run_details = cursor.fetchone()
            if not run_details:
                self.details_display.setText("Run record not found")
                return

            m, n, k, j, s, f, algorithm_run_id, timestamp, sample_count, execution_time = run_details[:10]

            if has_new_columns:
                run_count = run_details[10]
                formatted_id = run_details[11]
                algorithm = run_details[12]
            else:
                run_count = 1
                formatted_id = None
                algorithm = run_details[10]

            # Get samples
            cursor.execute('''
            SELECT sample FROM samples WHERE run_id = ? ORDER BY sample
            ''', (run_id,))

            samples = [row[0] for row in cursor.fetchall()]

            # Get results
            cursor.execute('''
            SELECT group_id, sample 
            FROM results 
            WHERE run_id = ? 
            ORDER BY group_id, sample
            ''', (run_id,))

            results = cursor.fetchall()

            # Organize results
            groups = {}
            for group_id, sample in results:
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(sample)

            # Use the formatted ID if available, otherwise generate one
            if formatted_id:
                run_format = formatted_id
            else:
                # For backward compatibility with old records
                run_format = f"{m}-{n}-{k}-{j}-{s}-{run_count}-{len(groups)}"

            # Create simplified HTML output with appropriate ID
            html_content = """
            <h2 style="text-align:center;">{}</h2>
            <p style="text-align:center;">Run Time: {}</p>
            <p style="text-align:center;">Execution Time: {:.2f} seconds</p>

            <h3 style="border-bottom:1px solid #cccccc; padding-bottom:5px;">Algorithm Parameters</h3>
            <table width="100%">
                <tr><td style="text-align:right;">Total Sample Number (m):</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Selected Sample Number (n):</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Combination Size (k):</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Subset Parameter (j):</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Coverage Parameter (s):</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Coverage Times (f):</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Algorithm:</td><td style="font-weight:bold;">{}</td></tr>
                <tr><td style="text-align:right;">Run Count:</td><td style="font-weight:bold;">{}</td></tr>
            </table>

            <h3 style="border-bottom:1px solid #cccccc; padding-bottom:5px;">Selected Sample Set</h3>
            <div style="background-color:#f9f9f9; padding:10px; border-radius:4px;">{}</div>

            <h3 style="border-bottom:1px solid #cccccc; padding-bottom:5px;">Optimal Combination Results ({} combinations)</h3>
            """.format(
                run_format, timestamp, execution_time, m, n, k, j, s, f, algorithm, run_count,
                ', '.join(samples), len(groups)
            )

            # Add combinations
            for group_id, group_samples in sorted(groups.items()):
                html_content += """
                <div style="margin:5px 0; padding:8px; background-color:#f9f9f9; border-radius:4px; border-left:3px solid #2196f3;">
                    <span style="font-weight:bold;">Combination {}:</span> 
                    <span>{}</span>
                </div>
                """.format(group_id, ', '.join(group_samples))

            # Set HTML content
            self.details_display.setHtml(html_content)
        except Exception as e:
            self.details_display.setText(f"Error loading results: {str(e)}")
        finally:
            if conn:
                conn.close()

    def delete_selected_run(self):
        """Delete selected run record"""
        selected_items = self.runs_list.selectedItems()
        if not selected_items:
            return

        # Get selected run ID
        run_id = selected_items[0].data(Qt.UserRole)

        # Confirm deletion
        reply = QMessageBox.question(self, "Confirm Delete",
                                     "Are you sure you want to delete the selected run record?",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply != QMessageBox.Yes:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign key constraints, ensure cascading delete
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()

            # Because foreign key constraints are already set, just delete main table record
            # Subtable records will be automatically cascaded deleted
            cursor.execute("DELETE FROM runs WHERE id = ?", (run_id,))

            conn.commit()

            # Update run record list
            self.load_saved_runs()

            # Clear details display
            self.details_display.clear()
            self.delete_btn.setEnabled(False)
            self.export_btn.setEnabled(False)

            QMessageBox.information(self, "Delete Successful", "Run record has been successfully deleted")
        except Exception as e:
            QMessageBox.warning(self, "Delete Failed", f"Error deleting run record: {str(e)}")
        finally:
            if conn:
                conn.close()

    def export_results(self):
        """Export results as text file"""
        selected_items = self.runs_list.selectedItems()
        if not selected_items:
            return

        # Get selected run ID
        run_id = selected_items[0].data(Qt.UserRole)

        # Choose save path
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Export Results", "", "Text Files (*.txt);;All Files (*.*)")

        if not file_path:
            return

        try:
            conn = sqlite3.connect(self.db_path)
            # Enable foreign key constraints
            conn.execute("PRAGMA foreign_keys = ON")
            cursor = conn.cursor()

            # Get run details
            cursor.execute('''
            SELECT m, n, k, j, s, f, run_id, timestamp, sample_count, execution_time
            FROM runs
            WHERE id = ?
            ''', (run_id,))

            run_details = cursor.fetchone()
            if not run_details:
                QMessageBox.warning(self, "Export Error", "Run record not found")
                return

            m, n, k, j, s, f, algorithm_run_id, timestamp, sample_count, execution_time = run_details

            # Get samples
            cursor.execute('''
            SELECT sample FROM samples WHERE run_id = ? ORDER BY sample
            ''', (run_id,))

            samples = [row[0] for row in cursor.fetchall()]

            # Get results
            cursor.execute('''
            SELECT group_id, sample 
            FROM results 
            WHERE run_id = ? 
            ORDER BY group_id, sample
            ''', (run_id,))

            results = cursor.fetchall()

            # Organize results
            groups = {}
            for group_id, sample in results:
                if group_id not in groups:
                    groups[group_id] = []
                groups[group_id].append(sample)

            # Format: m-n-k-j-s-x-y
            run_format = f"{m}-{n}-{k}-{j}-{s}-{algorithm_run_id}-{len(groups)}"

            # Write to file
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"===== {run_format} =====\n\n")
                f.write(f"Run Time: {timestamp}\n")
                f.write(f"Execution Time: {execution_time:.2f} seconds\n\n")
                f.write("Parameter Settings:\n")
                f.write(f"  m = {m} (Total Samples)\n")
                f.write(f"  n = {n} (Selected Samples)\n")
                f.write(f"  k = {k} (Combination Size)\n")
                f.write(f"  j = {j} (Subset Parameter)\n")
                f.write(f"  s = {s} (Coverage Parameter)\n")
                f.write(f"  f = {f} (Coverage Times)\n\n")
                f.write(f"Sample Set: {', '.join(samples)}\n\n")
                f.write(f"Found {len(groups)} optimal combinations:\n")

                for group_id, group_samples in sorted(groups.items()):
                    f.write(f"Combination {group_id}: {', '.join(group_samples)}\n")

            QMessageBox.information(self, "Export Successful", f"Results exported to: {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting results: {str(e)}")
        finally:
            if conn:
                conn.close()

    def clear_input_and_results(self):
        """Clear input fields and calculation results"""
        # Reset parameters to default values
        self.m_input.setValue(45)
        self.n_input.setValue(7)
        self.k_input.setValue(6)
        self.j_input.setValue(4)
        self.s_input.setValue(4)
        self.f_input.setValue(1)

        # Reset sample selection
        self.random_select.setChecked(True)
        self.sample_input.clear()
        self.samples_display.clear()

        # Clear results
        self.result_display.clear()
        self.current_results = []

        # Clear ID label
        self.id_label.setText("")

        # Reset buttons
        self.stop_btn.setEnabled(False)
        self.save_btn.setEnabled(False)
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Execute")

    def update_n_dependent_ui(self, n_value):
        """Update UI elements that depend on n value."""
        # Update k and s inputs if they exist
        if hasattr(self, 'k_input'):
            self.k_input.setMaximum(n_value)
        if hasattr(self, 's_input'):
            self.s_input.setMaximum(n_value)

    def on_algorithm_changed(self, index):
        """处理算法选择变化"""
        algorithm_names = ["Genetic Algorithm", "Simulated Annealing", "Greedy Algorithm"]
        if index >= 0 and index < len(algorithm_names):
            self.result_display.clear()
            self.result_display.append(f"已选择: {algorithm_names[index]}")

            # 清除之前的结果
            self.current_results = []
            self.save_btn.setEnabled(False)

    def setup_comparison_tab(self, tab):
        """Set up the algorithm comparison tab"""
        # Create main layout
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(15, 15, 15, 15)
        layout.setSpacing(20)

        # Parameter setting group
        param_group = QGroupBox("Algorithm Parameter Setting")
        param_layout = QGridLayout()
        param_layout.setSpacing(15)
        param_layout.setContentsMargins(20, 30, 20, 20)

        # Create label style
        label_style = """
            QLabel {
                font-size: 11pt;
                color: #333333;
                font-weight: normal;
            }
        """

        # Create input fields - First row
        # m parameter
        m_label = QLabel("m (45<=m<=54):")
        m_label.setStyleSheet(label_style)
        m_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.comp_m_input = QSpinBox()
        self.comp_m_input.setRange(45, 54)
        self.comp_m_input.setValue(45)
        self.comp_m_input.setMinimumWidth(80)  # 增加宽度
        param_layout.addWidget(m_label, 0, 0)
        param_layout.addWidget(self.comp_m_input, 0, 1)

        # n parameter
        n_label = QLabel("n (7<=n<=25):")
        n_label.setStyleSheet(label_style)
        n_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.comp_n_input = QSpinBox()
        self.comp_n_input.setRange(7, 25)
        self.comp_n_input.setValue(7)
        self.comp_n_input.setMinimumWidth(80)  # 增加宽度
        param_layout.addWidget(n_label, 0, 2)
        param_layout.addWidget(self.comp_n_input, 0, 3)

        # k parameter
        k_label = QLabel("k (4<=k<=7):")
        k_label.setStyleSheet(label_style)
        k_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.comp_k_input = QSpinBox()
        self.comp_k_input.setRange(4, 7)
        self.comp_k_input.setValue(6)  # Default value is 6
        self.comp_k_input.setMinimumWidth(80)  # 增加宽度
        param_layout.addWidget(k_label, 0, 4)
        param_layout.addWidget(self.comp_k_input, 0, 5)

        # Second row
        # j parameter
        j_label = QLabel("j (s<=j<=k):")
        j_label.setStyleSheet(label_style)
        j_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.comp_j_input = QSpinBox()
        self.comp_j_input.setRange(3, 25)
        self.comp_j_input.setValue(4)
        self.comp_j_input.setMinimumWidth(80)  # 增加宽度
        param_layout.addWidget(j_label, 1, 0)
        param_layout.addWidget(self.comp_j_input, 1, 1)

        # s parameter
        s_label = QLabel("s (3<=s<=7):")
        s_label.setStyleSheet(label_style)
        s_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.comp_s_input = QSpinBox()
        self.comp_s_input.setRange(3, 7)
        self.comp_s_input.setValue(4)
        self.comp_s_input.setMinimumWidth(80)  # 增加宽度
        param_layout.addWidget(s_label, 1, 2)
        param_layout.addWidget(self.comp_s_input, 1, 3)

        # f parameter
        f_label = QLabel("at least f s samples:")
        f_label.setStyleSheet(label_style)
        f_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.comp_f_input = QSpinBox()
        self.comp_f_input.setRange(1, 10)
        self.comp_f_input.setValue(1)
        self.comp_f_input.setMinimumWidth(80)  # 增加宽度
        param_layout.addWidget(f_label, 1, 4)
        param_layout.addWidget(self.comp_f_input, 1, 5)

        # 添加验证设置
        validate_label = QLabel("varify times:")
        validate_label.setStyleSheet(label_style)
        validate_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.validate_iterations = QSpinBox()
        self.validate_iterations.setRange(10, 100)
        self.validate_iterations.setValue(30)
        self.validate_iterations.setMinimumWidth(80)
        param_layout.addWidget(validate_label, 2, 0)
        param_layout.addWidget(self.validate_iterations, 2, 1)

        # 验证开关
        self.validate_checkbox = QRadioButton("use confidence")
        self.validate_checkbox.setChecked(True)
        param_layout.addWidget(self.validate_checkbox, 2, 2, 1, 2)

        # Set parameter constraints
        self.comp_m_input.valueChanged.connect(self.update_comp_n_max)
        self.comp_n_input.valueChanged.connect(self.update_comp_constraints)
        self.comp_j_input.valueChanged.connect(self.update_comp_constraints)
        self.comp_s_input.valueChanged.connect(self.update_comp_constraints)

        # 增加数字显示的样式
        spinbox_style = """
            QSpinBox {
                font-size: 12pt;
                padding: 4px;
                background-color: white;
                border: 1px solid #cccccc;
                border-radius: 4px;
            }
            QSpinBox::up-button, QSpinBox::down-button {
                width: 20px;
            }
        """

        # 应用样式到所有SpinBox
        self.comp_m_input.setStyleSheet(spinbox_style)
        self.comp_n_input.setStyleSheet(spinbox_style)
        self.comp_k_input.setStyleSheet(spinbox_style)
        self.comp_j_input.setStyleSheet(spinbox_style)
        self.comp_s_input.setStyleSheet(spinbox_style)
        self.comp_f_input.setStyleSheet(spinbox_style)
        self.validate_iterations.setStyleSheet(spinbox_style)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Top selection and button area
        top_controls = QWidget()
        top_layout = QHBoxLayout(top_controls)

        # Selection method (left side)
        selection_frame = QFrame()
        selection_layout = QHBoxLayout(selection_frame)
        selection_layout.setContentsMargins(0, 0, 0, 0)

        self.comp_random_select = QRadioButton("Random N")
        self.comp_random_select.setChecked(True)

        self.comp_manual_select = QRadioButton("Input N")

        selection_layout.addWidget(self.comp_random_select)
        selection_layout.addWidget(self.comp_manual_select)
        selection_layout.addStretch()

        top_layout.addWidget(selection_frame)
        top_layout.addStretch()

        # Buttons (right side)
        buttons_frame = QFrame()
        buttons_layout = QHBoxLayout(buttons_frame)
        buttons_layout.setContentsMargins(0, 0, 0, 0)
        buttons_layout.setSpacing(10)

        self.comp_start_btn = QPushButton("Start Comparison")
        self.comp_start_btn.clicked.connect(self.start_algorithm_comparison)
        buttons_layout.addWidget(self.comp_start_btn)

        self.comp_stop_btn = QPushButton("Stop")
        self.comp_stop_btn.clicked.connect(self.stop_comparison)
        self.comp_stop_btn.setEnabled(False)
        buttons_layout.addWidget(self.comp_stop_btn)

        self.comp_export_btn = QPushButton("Export")
        self.comp_export_btn.clicked.connect(self.export_comparison_results)
        self.comp_export_btn.setEnabled(False)
        buttons_layout.addWidget(self.comp_export_btn)

        self.comp_clear_btn = QPushButton("Clear")
        self.comp_clear_btn.clicked.connect(self.clear_comparison)
        buttons_layout.addWidget(self.comp_clear_btn)

        top_layout.addWidget(buttons_frame)

        layout.addWidget(top_controls)

        # User input area for manual entry
        user_input_group = QGroupBox("User Input")
        user_input_layout = QVBoxLayout(user_input_group)

        # Manual input area with custom styling
        manual_input_layout = QHBoxLayout()
        self.comp_sample_input = QLineEdit()
        self.comp_sample_input.setPlaceholderText("Enter sample numbers separated by commas (e.g., 01,02,03)")
        self.comp_sample_input.setEnabled(False)
        manual_input_layout.addWidget(self.comp_sample_input)

        user_input_layout.addLayout(manual_input_layout)

        # Connect signals
        self.comp_random_select.toggled.connect(self.toggle_comp_input_mode)
        self.comp_manual_select.toggled.connect(self.toggle_comp_input_mode)
        self.comp_n_input.valueChanged.connect(self.update_comp_n_dependent_ui)
        self.comp_n_input.valueChanged.connect(self.populate_comp_default_samples)

        layout.addWidget(user_input_group)

        # Progress bars for each algorithm
        progress_group = QGroupBox("Progress")
        progress_layout = QGridLayout()

        # Genetic Algorithm progress
        ga_label = QLabel("Genetic Algorithm:")
        ga_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.ga_progress = QProgressBar()
        self.ga_progress.setRange(0, 100)
        self.ga_progress.setValue(0)
        self.ga_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #4CAF50;
                width: 10px;
                margin: 0.5px;
            }
        """)
        progress_layout.addWidget(ga_label, 0, 0)
        progress_layout.addWidget(self.ga_progress, 0, 1)

        # Simulated Annealing progress
        sa_label = QLabel("Simulated Annealing:")
        sa_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.sa_progress = QProgressBar()
        self.sa_progress.setRange(0, 100)
        self.sa_progress.setValue(0)
        self.sa_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #2196F3;
                width: 10px;
                margin: 0.5px;
            }
        """)
        progress_layout.addWidget(sa_label, 1, 0)
        progress_layout.addWidget(self.sa_progress, 1, 1)

        # Greedy Algorithm progress
        greedy_label = QLabel("Greedy Algorithm:")
        greedy_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.greedy_progress = QProgressBar()
        self.greedy_progress.setRange(0, 100)
        self.greedy_progress.setValue(0)
        self.greedy_progress.setStyleSheet("""
            QProgressBar {
                border: 1px solid #cccccc;
                border-radius: 4px;
                text-align: center;
                height: 20px;
                background-color: #f5f5f5;
            }
            QProgressBar::chunk {
                background-color: #FF9800;
                width: 10px;
                margin: 0.5px;
            }
        """)
        progress_layout.addWidget(greedy_label, 2, 0)
        progress_layout.addWidget(self.greedy_progress, 2, 1)

        progress_group.setLayout(progress_layout)
        layout.addWidget(progress_group)

        # Results table
        self.results_table = QTableWidget()
        self.results_table.setColumnCount(6)  # 增加了置信度和相对置信度列
        self.results_table.setHorizontalHeaderLabels(
            ["Algorithm", "Execution Time", "# Combinations", "Confidence", "Relative Confidence", "Combinations"])
        self.results_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)  # 更新为新的列索引
        self.results_table.setMinimumHeight(200)
        layout.addWidget(self.results_table)

    def update_comp_n_max(self):
        """Update the maximum allowed value for n input based on m in comparison tab."""
        m_value = self.comp_m_input.value()
        # n can't be larger than m
        self.comp_n_input.setMaximum(m_value)
        # If current n is larger than m, adjust it
        if self.comp_n_input.value() > m_value:
            self.comp_n_input.setValue(m_value)

        # Update UI elements that depend on n
        n_value = self.comp_n_input.value()
        self.update_comp_n_dependent_ui(n_value)

        # Update default samples if manual selection is enabled
        if hasattr(self, 'comp_manual_select') and self.comp_manual_select.isChecked():
            self.populate_comp_default_samples(n_value)

    def update_comp_constraints(self):
        """Update j, s, k constraints for comparison tab"""
        n_value = self.comp_n_input.value()
        j_value = self.comp_j_input.value()
        s_value = self.comp_s_input.value()

        # Ensure j >= s
        if j_value < s_value:
            self.comp_j_input.setValue(s_value)

        # Ensure j <= n, k <= n
        self.comp_j_input.setMaximum(n_value)
        self.comp_k_input.setMaximum(min(7, n_value))

        # Ensure s <= j, s <= k
        s_max = min(self.comp_j_input.value(), self.comp_k_input.value())
        self.comp_s_input.setMaximum(min(7, s_max))

    def update_comp_n_dependent_ui(self, n_value):
        """Update UI elements that depend on n value in comparison tab."""
        # Update k and s inputs if they exist
        if hasattr(self, 'comp_k_input'):
            self.comp_k_input.setMaximum(n_value)
        if hasattr(self, 'comp_s_input'):
            self.comp_s_input.setMaximum(n_value)

    def toggle_comp_input_mode(self):
        """Toggle between random and manual input modes in comparison tab."""
        if self.comp_manual_select.isChecked():
            self.comp_sample_input.setEnabled(True)
            # Populate default samples when switching to manual mode
            self.populate_comp_default_samples(self.comp_n_input.value())
        else:
            self.comp_sample_input.setEnabled(False)
            self.comp_sample_input.setText("")

    def populate_comp_default_samples(self, n_value):
        """Generate a default sequential list of samples based on n value for comparison tab."""
        if n_value <= 0:
            self.comp_sample_input.setText("")
            return

        # Generate samples with leading zeros for single digits (01, 02, etc.)
        samples = [f"{i:02d}" for i in range(1, n_value + 1)]
        sample_text = ", ".join(samples)
        self.comp_sample_input.setText(sample_text)

    def start_algorithm_comparison(self):
        """Start the comparison of all three algorithms simultaneously."""
        # Get parameter values
        m = self.comp_m_input.value()
        n = self.comp_n_input.value()
        k = self.comp_k_input.value()
        j = self.comp_j_input.value()
        s = self.comp_s_input.value()
        f = self.comp_f_input.value()

        # Validate parameters
        if not (3 <= s <= j <= k <= n <= m):
            QMessageBox.warning(self, "Parameter Error",
                                "Parameters must satisfy: 3 ≤ s ≤ j ≤ k ≤ n ≤ m")
            return

        # Prepare samples
        self.comp_samples = []
        if self.comp_random_select.isChecked():
            # Randomly select samples
            all_samples = [f"{i:02d}" for i in range(1, m + 1)]
            self.comp_samples = random.sample(all_samples, n)
        else:
            # Manual input samples
            input_text = self.comp_sample_input.text().strip()
            if not input_text:
                QMessageBox.warning(self, "Input Error", "Please enter sample numbers")
                return

            try:
                input_samples = input_text.split(',')
                self.comp_samples = [s.strip().zfill(2) for s in input_samples]

                # Validate samples
                if len(self.comp_samples) != n:
                    QMessageBox.warning(self, "Input Error", f"Please enter exactly {n} samples")
                    return

                if len(set(self.comp_samples)) != len(self.comp_samples):
                    QMessageBox.warning(self, "Input Error", "Sample numbers cannot be duplicated")
                    return

                if any(not s.isdigit() or int(s) < 1 or int(s) > m for s in self.comp_samples):
                    QMessageBox.warning(self, "Input Error", f"Sample numbers must be between 01-{m:02d}")
                    return
            except:
                QMessageBox.warning(self, "Input Error", "Please enter valid sample numbers, separated by commas")
                return

        # 排序样本但不显示它们
        self.comp_samples.sort()

        # 删除以下代码
        # samples_with_numbers = []
        # for i, sample in enumerate(self.comp_samples, 1):
        #    samples_with_numbers.append(f"{i}# {sample}")
        # self.comp_samples_display.setText("\n".join(samples_with_numbers))

        # Update UI before starting
        self.comp_start_btn.setEnabled(False)
        self.comp_stop_btn.setEnabled(True)

        # Reset progress bars
        self.ga_progress.setValue(0)
        self.sa_progress.setValue(0)
        self.greedy_progress.setValue(0)

        # Clear results table
        self.results_table.setRowCount(0)

        # Create and start threads
        self.ga_thread = ComputationThread(self.comp_samples, j, s, k, f, "genetic_algorithm")
        self.sa_thread = ComputationThread(self.comp_samples, j, s, k, f, "simulated_annealing")
        self.greedy_thread = ComputationThread(self.comp_samples, j, s, k, f, "greedy_algorithm")

        # Connect signals
        self.ga_thread.result_ready.connect(
            lambda result, time: self.handle_comparison_result("Genetic Algorithm", result, time))
        self.sa_thread.result_ready.connect(
            lambda result, time: self.handle_comparison_result("Simulated Annealing", result, time))
        self.greedy_thread.result_ready.connect(
            lambda result, time: self.handle_comparison_result("Greedy Algorithm", result, time))

        self.ga_thread.progress_value.connect(self.ga_progress.setValue)
        self.sa_thread.progress_value.connect(self.sa_progress.setValue)
        self.greedy_thread.progress_value.connect(self.greedy_progress.setValue)

        # Start threads
        self.ga_thread.start()
        self.sa_thread.start()
        self.greedy_thread.start()

    def handle_comparison_result(self, algorithm_name, results, execution_time):
        """Handle results from algorithm comparison threads."""
        # 存储算法结果用于验证
        if not hasattr(self, 'algorithm_results'):
            self.algorithm_results = {}

        self.algorithm_results[algorithm_name] = {
            'results': results,
            'execution_time': execution_time
        }

        # 检查所有算法是否都已完成
        if (hasattr(self, 'ga_thread') and not self.ga_thread.isRunning() and
                hasattr(self, 'sa_thread') and not self.sa_thread.isRunning() and
                hasattr(self, 'greedy_thread') and not self.greedy_thread.isRunning()):

            # 如果启用了置信度验证，则进行验证
            if self.validate_checkbox.isChecked():
                self.validate_all_results()
            else:
                # 不进行验证，直接显示结果
                self.display_comparison_results()

            # 恢复UI状态
            self.comp_start_btn.setEnabled(True)
            self.comp_stop_btn.setEnabled(False)

    def validate_all_results(self):
        """对所有算法结果进行置信度验证"""
        # 创建验证器
        j = self.comp_j_input.value()
        s = self.comp_s_input.value()
        k = self.comp_k_input.value()
        f = self.comp_f_input.value()
        iterations = self.validate_iterations.value()

        # 在新线程中运行验证，避免阻塞UI
        self.validation_thread = ValidationThread(
            self.comp_samples,
            self.algorithm_results,
            j, s, k, f,
            iterations
        )
        self.validation_thread.validation_complete.connect(self.display_validation_results)
        self.validation_thread.progress_update.connect(self.update_validation_progress)
        self.validation_thread.error_occurred.connect(self.handle_error)  # 连接错误信号

        # 开始验证
        self.results_table.setRowCount(0)
        for i in range(len(self.algorithm_results)):
            self.results_table.insertRow(i)

        # 更新UI
        for i, (alg_name, data) in enumerate(self.algorithm_results.items()):
            self.results_table.setItem(i, 0, QTableWidgetItem(alg_name))
            self.results_table.setItem(i, 1, QTableWidgetItem(f"{data['execution_time']:.4f} sec"))
            self.results_table.setItem(i, 2, QTableWidgetItem(str(len(data['results']))))
            # 置信度列设置为"计算中..."
            self.results_table.setItem(i, 3, QTableWidgetItem("计算中..."))
            self.results_table.setItem(i, 4, QTableWidgetItem("计算中..."))

            # 组合列
            combinations_text = ""
            for j, group in enumerate(data['results'], 1):
                if j > 1:
                    combinations_text += "\n"
                combinations_text += f"{j}# {', '.join(group)}"
            combinations_item = QTableWidgetItem(combinations_text)
            combinations_item.setToolTip(combinations_text)
            self.results_table.setItem(i, 5, combinations_item)
            self.results_table.setRowHeight(i, 100)

        # 启动验证线程
        self.validation_thread.start()

    def update_validation_progress(self, message):
        """更新验证进度"""
        self.statusBar().showMessage(message)

    def display_validation_results(self, validation_results):
        """显示验证结果"""
        # 更新表格中的置信度数据
        for i, (alg_name, result) in enumerate(validation_results.items()):
            # 找到对应算法所在行
            for row in range(self.results_table.rowCount()):
                if self.results_table.item(row, 0).text() == alg_name:
                    # 更新置信度和相对置信度
                    confidence = result.get("confidence", 0.0)
                    rel_confidence = result.get("relative_confidence", 0.0)

                    confidence_item = QTableWidgetItem(f"{confidence:.4f}")
                    rel_confidence_item = QTableWidgetItem(f"{rel_confidence:.4f}")

                    # 设置单元格对齐方式
                    confidence_item.setTextAlignment(Qt.AlignCenter)
                    rel_confidence_item.setTextAlignment(Qt.AlignCenter)

                    # 根据置信度值设置颜色
                    if confidence >= 0.8:
                        confidence_item.setBackground(QColor(200, 255, 200))  # 绿色
                    elif confidence >= 0.6:
                        confidence_item.setBackground(QColor(255, 255, 200))  # 黄色
                    else:
                        confidence_item.setBackground(QColor(255, 200, 200))  # 红色

                    # 设置相对置信度的颜色
                    if rel_confidence >= 0.9:
                        rel_confidence_item.setBackground(QColor(200, 255, 200))  # 绿色
                    elif rel_confidence >= 0.7:
                        rel_confidence_item.setBackground(QColor(255, 255, 200))  # 黄色
                    else:
                        rel_confidence_item.setBackground(QColor(255, 200, 200))  # 红色

                    # 更新表格
                    self.results_table.setItem(row, 3, confidence_item)
                    self.results_table.setItem(row, 4, rel_confidence_item)

                    # 设置行背景色
                    if "Genetic" in alg_name:
                        color = QColor(240, 255, 240)
                    elif "Simulated" in alg_name:
                        color = QColor(240, 248, 255)
                    else:
                        color = QColor(255, 248, 240)

                    # 只设置算法名称和组合列的背景色
                    self.results_table.item(row, 0).setBackground(color)
                    self.results_table.item(row, 5).setBackground(color)

                    break

        # 按置信度排序
        self.results_table.sortItems(4, Qt.DescendingOrder)

        # 清空状态栏
        self.statusBar().clearMessage()

        # 启用导出按钮
        self.comp_export_btn.setEnabled(True)

        # 找出置信度最高的算法
        best_row = 0
        best_alg = self.results_table.item(best_row, 0).text()
        best_confidence = float(self.results_table.item(best_row, 4).text())

        QMessageBox.information(self, "complete",
                                f"Complete！\nThe best is: {best_alg}\nConfidence: {best_confidence:.4f}")

    def display_comparison_results(self):
        """不进行验证，直接显示比较结果"""
        # 清空表格
        self.results_table.setRowCount(0)

        # 添加算法结果
        for alg_name, data in self.algorithm_results.items():
            row_position = self.results_table.rowCount()
            self.results_table.insertRow(row_position)

            # 算法名称
            name_item = QTableWidgetItem(alg_name)
            name_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_position, 0, name_item)

            # 执行时间
            time_item = QTableWidgetItem(f"{data['execution_time']:.4f} sec")
            time_item.setTextAlignment(Qt.AlignCenter)
            time_item.setData(Qt.UserRole, data['execution_time'])
            self.results_table.setItem(row_position, 1, time_item)

            # 组合数量
            count_item = QTableWidgetItem(str(len(data['results'])))
            count_item.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_position, 2, count_item)

            # 置信度和相对置信度列设置为N/A
            na_item1 = QTableWidgetItem("N/A")
            na_item1.setTextAlignment(Qt.AlignCenter)
            na_item2 = QTableWidgetItem("N/A")
            na_item2.setTextAlignment(Qt.AlignCenter)
            self.results_table.setItem(row_position, 3, na_item1)
            self.results_table.setItem(row_position, 4, na_item2)

            # 组合结果
            combinations_text = ""
            for i, group in enumerate(data['results'], 1):
                if i > 1:
                    combinations_text += "\n"
                combinations_text += f"{i}# {', '.join(group)}"

            combinations_item = QTableWidgetItem(combinations_text)
            combinations_item.setToolTip(combinations_text)
            self.results_table.setItem(row_position, 5, combinations_item)

            # 设置行高
            self.results_table.setRowHeight(row_position, 100)

            # 使用不同背景色标识不同算法
            if "Genetic" in alg_name:
                color = QColor(240, 255, 240)
            elif "Simulated" in alg_name:
                color = QColor(240, 248, 255)
            else:
                color = QColor(255, 248, 240)

            # 设置行背景色
            for col in [0, 5]:  # 只设置算法名称和组合列的背景色
                self.results_table.item(row_position, col).setBackground(color)

        # 按执行时间排序
        self.results_table.sortItems(1, Qt.AscendingOrder)

        # 启用导出按钮
        self.comp_export_btn.setEnabled(True)

        # 完成后显示比较结果
        fastest_alg = self.results_table.item(0, 0).text()
        QMessageBox.information(self, "Comparison Complete",
                                f"All algorithms finished.\nFastest: {fastest_alg}")

    def stop_comparison(self):
        """Stop all comparison threads."""
        if hasattr(self, 'ga_thread') and self.ga_thread.isRunning():
            self.ga_thread.stop()

        if hasattr(self, 'sa_thread') and self.sa_thread.isRunning():
            self.sa_thread.stop()

        if hasattr(self, 'greedy_thread') and self.greedy_thread.isRunning():
            self.greedy_thread.stop()

        self.comp_start_btn.setEnabled(True)
        self.comp_stop_btn.setEnabled(False)

    def clear_comparison(self):
        """Clear all comparison inputs and results."""
        # Reset parameters to default values
        self.comp_m_input.setValue(45)
        self.comp_n_input.setValue(7)
        self.comp_k_input.setValue(6)
        self.comp_j_input.setValue(4)
        self.comp_s_input.setValue(4)
        self.comp_f_input.setValue(1)

        # Reset sample selection
        self.comp_random_select.setChecked(True)
        self.comp_sample_input.clear()

        # Clear results table
        self.results_table.setRowCount(0)

        # Reset progress bars
        self.ga_progress.setValue(0)
        self.sa_progress.setValue(0)
        self.greedy_progress.setValue(0)

        # 禁用导出按钮
        self.comp_export_btn.setEnabled(False)

    def export_comparison_results(self):
        """导出算法比较结果为文本文件或PDF文件"""
        # 检查是否有结果
        if self.results_table.rowCount() == 0:
            return

        # 确定文件过滤器
        file_filter = "Text Files (*.txt)"
        if PDF_AVAILABLE:
            file_filter = "PDF Files (*.pdf);;Text Files (*.txt);;All Files (*.*)"
        else:
            file_filter = "Text Files (*.txt);;All Files (*.*)"

        # 选择保存路径
        file_path, selected_filter = QFileDialog.getSaveFileName(
            self, "Export Comparison Results", "", file_filter)

        if not file_path:
            return

        try:
            # 获取参数设置
            m = self.comp_m_input.value()
            n = self.comp_n_input.value()
            k = self.comp_k_input.value()
            j = self.comp_j_input.value()
            s = self.comp_s_input.value()
            f = self.comp_f_input.value()
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            has_confidence = self.validate_checkbox.isChecked()

            # 如果是PDF格式，询问是否添加AI总结
            ai_summary = ""
            if file_path.lower().endswith('.pdf') and PDF_AVAILABLE:
                reply = QMessageBox.question(
                    self, "Summary",
                    "Would you like to add an analysis summary to your PDF?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.Yes
                )

                if reply == QMessageBox.Yes:
                    ai_summary = self._get_ai_summary(m, n, k, j, s, f, has_confidence)

                # 导出带有AI总结的PDF
                self._export_to_pdf(file_path, m, n, k, j, s, f, timestamp, has_confidence, ai_summary)
            else:
                self._export_to_text(file_path, m, n, k, j, s, f, timestamp, has_confidence)

            QMessageBox.information(self, "Export Successful", f"Comparison results exported to: {file_path}")
        except Exception as e:
            QMessageBox.warning(self, "Export Error", f"Error exporting comparison results: {str(e)}")
            if file_path.lower().endswith('.pdf') and not PDF_AVAILABLE:
                QMessageBox.information(self, "PDF Export",
                                        "PDF export requires the ReportLab library. Please install it using 'pip install reportlab'.")

    def _get_ai_summary(self, m, n, k, j, s, f, has_confidence):
        """获取AI对比较结果的总结"""
        # 创建输入对话框
        dialog = QDialog(self)
        dialog.setWindowTitle("Analysis Summary")
        dialog.setMinimumSize(600, 400)

        layout = QVBoxLayout(dialog)

        # 添加说明文本
        guide_label = QLabel(
            "Please enter or edit the analysis summary of algorithm comparison results below. This text will be added to the PDF report.",
            dialog)
        layout.addWidget(guide_label)

        # 总结输入区域
        summary_text = QPlainTextEdit(dialog)

        # 添加默认总结
        default_summary = self._generate_default_summary(has_confidence)
        summary_text.setPlainText(default_summary)
        layout.addWidget(summary_text)

        # 对话框按钮
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel, dialog)
        button_box.accepted.connect(dialog.accept)
        button_box.rejected.connect(dialog.reject)
        layout.addWidget(button_box)

        # 显示对话框
        if dialog.exec_() == QDialog.Accepted:
            return summary_text.toPlainText()
        else:
            return ""

    def _generate_default_summary(self, has_confidence):
        """生成默认的总结文本"""
        # 查找执行时间最短的算法
        fastest_alg = None
        fastest_time = float('inf')

        # 查找置信度最高的算法
        most_confident_alg = None
        highest_confidence = -1

        # 收集所有算法数据
        alg_data = []

        for row in range(self.results_table.rowCount()):
            alg_name = self.results_table.item(row, 0).text()

            # 解析执行时间
            time_text = self.results_table.item(row, 1).text()
            exec_time = float(time_text.split()[0])  # 提取秒数

            if exec_time < fastest_time:
                fastest_time = exec_time
                fastest_alg = alg_name

            # 存储组合数量
            comb_count = int(self.results_table.item(row, 2).text())

            # 检查是否有置信度数据
            confidence_value = 0
            if has_confidence:
                confidence_text = self.results_table.item(row, 4).text()
                if confidence_text != "N/A":
                    try:
                        confidence_value = float(confidence_text)
                        if confidence_value > highest_confidence:
                            highest_confidence = confidence_value
                            most_confident_alg = alg_name
                    except ValueError:
                        pass

            alg_data.append((alg_name, exec_time, comb_count, confidence_value))

        # 生成总结文本
        summary = "Algorithm Comparison Analysis:\n\n"

        # 执行时间分析
        if fastest_alg:
            summary += f"Efficiency Analysis: {fastest_alg} demonstrates the highest efficiency with an execution time of {fastest_time:.4f} seconds."

            # 计算与其他算法的时间比较
            for alg, time, _, _ in alg_data:
                if alg != fastest_alg:
                    time_diff_percent = ((time - fastest_time) / fastest_time) * 100
                    summary += f" In comparison, {alg} executed in {time:.4f} seconds, which is {time_diff_percent:.1f}% slower than the fastest algorithm."

        summary += "\n\n"

        # 置信度分析
        if has_confidence and most_confident_alg:
            summary += f"Reliability Analysis: {most_confident_alg} shows the highest relative confidence at {highest_confidence:.4f}, indicating superior result reliability."

            if most_confident_alg == fastest_alg:
                summary += f" This means {fastest_alg} excels in both efficiency and reliability, making it an optimal choice."
            else:
                summary += f" However, the most efficient algorithm is {fastest_alg}, suggesting a trade-off between efficiency and reliability."

        summary += "\n\n"

        # 组合数量分析
        summary += "Combination Generation Analysis: "
        most_combinations = max(alg_data, key=lambda x: x[2])
        least_combinations = min(alg_data, key=lambda x: x[2])

        if most_combinations[2] == least_combinations[2]:
            summary += f"All algorithms generated the same number of combinations ({most_combinations[2]})."
        else:
            summary += f"{most_combinations[0]} generated the most combinations ({most_combinations[2]}), while {least_combinations[0]} produced the least ({least_combinations[2]})."
            summary += f" This variation in combination count reflects different strategies employed by the algorithms in generating coverage sets."

        summary += "\n\n"

        # 总体建议
        summary += "Recommendations: "
        if has_confidence:
            if most_confident_alg == fastest_alg:
                summary += f"{fastest_alg} demonstrates the best overall performance, balancing both execution efficiency and result reliability. It is recommended as the preferred algorithm for this problem."
            else:
                summary += f"If prioritizing execution efficiency, {fastest_alg} is recommended; if result reliability is more important, {most_confident_alg} would be the better choice."
        else:
            summary += f"Based on execution efficiency considerations, {fastest_alg} is the most suitable algorithm for this problem."

        return summary

    def _export_to_pdf(self, file_path, m, n, k, j, s, f, timestamp, has_confidence, ai_summary=""):
        """导出为PDF格式"""
        try:
            from reportlab.lib import colors
            from reportlab.lib.pagesizes import letter
            from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
            from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
            from reportlab.lib.units import inch

            # 创建PDF文档对象
            doc = SimpleDocTemplate(file_path, pagesize=letter)

            # 创建样式表
            styles = getSampleStyleSheet()
            title_style = styles['Heading1']
            subtitle_style = styles['Heading2']
            normal_style = styles['Normal']

            # 自定义段落样式
            summary_style = ParagraphStyle(
                'SummaryStyle',
                parent=styles['Normal'],
                fontSize=11,
                leading=14,
                spaceAfter=6,
                textColor=colors.darkblue
            )

            # 添加标题和时间戳
            elements = [
                Paragraph("Algorithm Comparison Results", title_style),
                Paragraph(f"Export Time: {timestamp}", normal_style),
                Spacer(1, 0.2 * inch)
            ]

            # 如果有AI总结，添加到PDF
            if ai_summary:
                elements.append(Paragraph("AI Analysis Summary", subtitle_style))

                # 创建摘要的背景框
                summary_frame_style = TableStyle([
                    ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                    ('BACKGROUND', (0, 0), (-1, -1), colors.lightcyan),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ])

                # 将总结文本拆分为段落
                paragraphs = []
                for paragraph in ai_summary.split('\n\n'):
                    if paragraph.strip():
                        paragraphs.append(Paragraph(paragraph, summary_style))
                        paragraphs.append(Spacer(1, 0.1 * inch))

                # 创建一个只有一个单元格的表格，包含总结文本
                if paragraphs:
                    summary_table = Table([[paragraphs]], colWidths=[6.5 * inch])
                    summary_table.setStyle(summary_frame_style)
                    elements.append(summary_table)

                elements.append(Spacer(1, 0.3 * inch))

            # 添加参数设置
            elements.append(Paragraph("Parameter Settings", subtitle_style))

            # 主算法参数
            elements.append(Paragraph("Main Algorithm Parameters:", styles['Heading4']))
            param_data = [
                ["Parameter", "Value", "Description"],
                ["m", str(m), "Total Samples"],
                ["n", str(n), "Selected Samples"],
                ["k", str(k), "Combination Size"],
                ["j", str(j), "Subset Parameter"],
                ["s", str(s), "Coverage Parameter"],
                ["f", str(f), "Coverage Times"]
            ]

            param_table = Table(param_data, colWidths=[1.5 * inch, 1 * inch, 3 * inch])
            param_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'CENTER')
            ]))
            elements.append(param_table)
            elements.append(Spacer(1, 0.2 * inch))

            # 验证参数
            iterations = self.validate_iterations.value() if hasattr(self, 'validate_iterations') else "N/A"
            validation_enabled = "Yes" if self.validate_checkbox.isChecked() else "No" if hasattr(self,
                                                                                                  'validate_checkbox') else "N/A"

            elements.append(Paragraph("Validation Parameters:", styles['Heading4']))
            validation_data = [
                ["Parameter", "Value", "Description"],
                ["Confidence Validation", validation_enabled, "Whether confidence calculation is enabled"],
                ["Iterations", str(iterations), "Number of iterations for confidence calculation"],
            ]

            validation_table = Table(validation_data, colWidths=[1.5 * inch, 1 * inch, 3 * inch])
            validation_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (1, -1), 'CENTER')
            ]))
            elements.append(validation_table)
            elements.append(Spacer(1, 0.2 * inch))

            # 添加验证方法说明
            if has_confidence:
                elements.append(Paragraph("Validation Method:", styles['Heading4']))
                validation_explanation = (
                    "The confidence values are calculated by running multiple validation iterations "
                    f"({iterations} times) to verify the stability and reliability of the solutions generated by each algorithm. "
                    "For each iteration, the algorithm's solutions are tested against random test cases to measure their "
                    "coverage and effectiveness. Higher confidence values indicate more robust and reliable solutions."
                )
                elements.append(Paragraph(validation_explanation, normal_style))
                elements.append(Spacer(1, 0.3 * inch))

            # 添加样本集
            elements.append(Paragraph("Sample Set", subtitle_style))
            elements.append(Paragraph(', '.join(self.comp_samples), normal_style))
            elements.append(Spacer(1, 0.2 * inch))

            # 添加比较结果表格
            elements.append(Paragraph("Comparison Results", subtitle_style))

            # 表头
            header = ["Algorithm", "Execution Time", "# Combinations"]
            if has_confidence:
                header.extend(["Confidence", "Rel. Confidence"])

            # 准备表格数据
            table_data = [header]

            # 获取所有行的数据
            for row in range(self.results_table.rowCount()):
                row_data = []
                row_data.append(self.results_table.item(row, 0).text())
                row_data.append(self.results_table.item(row, 1).text())
                row_data.append(self.results_table.item(row, 2).text())
                if has_confidence:
                    row_data.append(self.results_table.item(row, 3).text())
                    row_data.append(self.results_table.item(row, 4).text())
                table_data.append(row_data)

            # 创建并格式化表格
            col_widths = [1.5 * inch, 1.3 * inch, 1.3 * inch]
            if has_confidence:
                col_widths.extend([1 * inch, 1.2 * inch])

            results_table = Table(table_data, colWidths=col_widths)

            table_style = [
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('GRID', (0, 0), (-1, -1), 1, colors.black),
                ('ALIGN', (1, 1), (-1, -1), 'CENTER')
            ]

            # 根据算法名称设置不同行的背景色
            for i in range(1, len(table_data)):
                if "Genetic" in table_data[i][0]:
                    table_style.append(('BACKGROUND', (0, i), (0, i), colors.lightgreen))
                elif "Simulated" in table_data[i][0]:
                    table_style.append(('BACKGROUND', (0, i), (0, i), colors.lightblue))
                else:
                    table_style.append(('BACKGROUND', (0, i), (0, i), colors.lightcoral))

            results_table.setStyle(TableStyle(table_style))
            elements.append(results_table)
            elements.append(Spacer(1, 0.3 * inch))

            # 添加算法描述
            elements.append(Paragraph("Algorithm Description", subtitle_style))

            algorithms_info = [
                {
                    "name": "Genetic Algorithm",
                    "description": "A method for solving optimization problems based on natural selection. The algorithm creates a population of potential solutions and improves them through mutation, crossover, and selection operations across multiple generations.",
                    "color": colors.lightgreen
                },
                {
                    "name": "Simulated Annealing",
                    "description": "An optimization technique inspired by annealing in metallurgy. It starts with a high 'temperature' allowing exploration of the solution space, then gradually 'cools down' to refine the solution and avoid local optima.",
                    "color": colors.lightblue
                },
                {
                    "name": "Greedy Algorithm",
                    "description": "A simple approach that makes locally optimal choices at each step with the hope of finding a global optimum. It builds a solution incrementally, choosing the next element that offers the most immediate benefit.",
                    "color": colors.lightcoral
                }
            ]

            for algo_info in algorithms_info:
                # 创建算法描述的背景框
                algo_name = algo_info["name"]
                algo_desc = algo_info["description"]
                algo_color = algo_info["color"]

                # 使用表格创建带颜色的标题
                header_data = [[Paragraph(algo_name, styles['Heading4'])]]
                header_table = Table(header_data, colWidths=[6.5 * inch])
                header_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, -1), algo_color),
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 6),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
                    ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(header_table)

                desc_data = [[Paragraph(algo_desc, normal_style)]]
                desc_table = Table(desc_data, colWidths=[6.5 * inch])
                desc_table.setStyle(TableStyle([
                    ('LEFTPADDING', (0, 0), (-1, -1), 12),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 12),
                    ('TOPPADDING', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                    ('BOX', (0, 0), (-1, -1), 1, colors.grey),
                ]))
                elements.append(desc_table)
                elements.append(Spacer(1, 0.1 * inch))

            elements.append(Spacer(1, 0.2 * inch))

            # 添加详细组合结果
            elements.append(Paragraph("Detailed Combinations", subtitle_style))

            # 为每个算法添加详细组合
            for row in range(self.results_table.rowCount()):
                alg_name = self.results_table.item(row, 0).text()
                combinations = self.results_table.item(row, 5).text()

                elements.append(Paragraph(alg_name, styles['Heading3']))

                # 分割组合文本并格式化
                combo_lines = combinations.split('\n')
                for line in combo_lines:
                    elements.append(Paragraph(line, normal_style))

                elements.append(Spacer(1, 0.2 * inch))

            # 生成PDF文档
            doc.build(elements)

            QMessageBox.information(self, "导出成功", f"结果已成功导出到PDF文件:\n{file_path}")
        except ImportError:
            QMessageBox.critical(self, "导出错误", "缺少必要的PDF导出模块(reportlab)，请确保已正确安装。")
            print("PDF导出错误: 缺少reportlab模块")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出PDF时发生错误:\n{str(e)}")
            print(f"PDF导出错误: {str(e)}")

    def _export_to_text(self, file_path, m, n, k, j, s, f, timestamp, has_confidence):
        """导出为文本格式"""
        try:
            with open(file_path, 'w', encoding='utf-8') as file:
                file.write(f"===== Algorithm Comparison Results =====\n")
                file.write(f"Export Time: {timestamp}\n\n")

                file.write("Parameter Settings:\n")
                file.write(f"  m = {m} (Total Samples)\n")
                file.write(f"  n = {n} (Selected Samples)\n")
                file.write(f"  k = {k} (Combination Size)\n")
                file.write(f"  j = {j} (Subset Parameter)\n")
                file.write(f"  s = {s} (Coverage Parameter)\n")
                file.write(f"  f = {f} (Coverage Times)\n\n")

                file.write(f"Sample Set: {', '.join(self.comp_samples)}\n\n")

                # 表格标题
                file.write(f"{'Algorithm':<20} {'Execution Time':<15} {'# Combinations':<15}")
                if has_confidence:
                    file.write(f" {'Confidence':<12} {'Rel. Confidence':<15}")
                file.write("\n")
                file.write("-" * (50 + (30 if has_confidence else 0)) + "\n")

                # 获取所有行的数据
                for row in range(self.results_table.rowCount()):
                    alg_name = self.results_table.item(row, 0).text()
                    exec_time = self.results_table.item(row, 1).text()
                    comb_count = self.results_table.item(row, 2).text()
                    confidence = self.results_table.item(row, 3).text() if self.results_table.item(row, 3) else "N/A"
                    rel_confidence = self.results_table.item(row, 4).text() if self.results_table.item(row,
                                                                                                       4) else "N/A"

                    file.write(f"{alg_name:<20} {exec_time:<15} {comb_count:<15}")
                    if has_confidence:
                        file.write(f" {confidence:<12} {rel_confidence:<15}")
                    file.write("\n")

                file.write("\n\nDetailed Combinations:\n")
                file.write("=" * 40 + "\n\n")

                # 输出每个算法的详细组合
                for row in range(self.results_table.rowCount()):
                    alg_name = self.results_table.item(row, 0).text()
                    combinations = self.results_table.item(row, 5).text() if self.results_table.item(row, 5) else ""

                    file.write(f"{alg_name}:\n")
                    file.write("-" * 40 + "\n")
                    file.write(f"{combinations}\n\n")

            QMessageBox.information(self, "导出成功", f"结果已成功导出到文本文件:\n{file_path}")
        except Exception as e:
            QMessageBox.critical(self, "导出错误", f"导出文件时发生错误:\n{str(e)}")
            print(f"导出文本文件错误: {str(e)}")


# 添加验证线程类
class ValidationThread(QThread):
    """用于在后台执行算法结果验证的线程"""
    validation_complete = pyqtSignal(dict)
    progress_update = pyqtSignal(str)
    error_occurred = pyqtSignal(str)  # 添加错误信号

    def __init__(self, samples, algorithm_results, j, s, k, f, iterations=30):
        super().__init__()
        self.samples = samples
        self.algorithm_results = algorithm_results
        self.j = j
        self.s = s
        self.k = k
        self.f = f
        self.iterations = iterations

    def run(self):
        """执行验证过程"""
        try:
            self.progress_update.emit("开始验证算法结果置信度...")

            # 准备算法结果字典
            solutions_dict = {alg_name: data['results'] for alg_name, data in self.algorithm_results.items()}

            # 创建验证器 - 禁用进度条
            # 修改tqdm导入，避免文件写入问题
            import tqdm
            tqdm.tqdm = lambda *args, **kwargs: args[0]  # 完全禁用tqdm进度条

            validator = SolutionValidator(self.samples, self.j, self.s, self.k, self.f)

            # 进行验证
            self.progress_update.emit(f"正在对 {len(solutions_dict)} 个算法结果进行验证 ({self.iterations} 次迭代)...")
            validation_results = validator.compare_solutions(solutions_dict)

            # 验证完成
            self.progress_update.emit("验证完成！")

            # 发送结果
            self.validation_complete.emit(validation_results)

        except Exception as e:
            import traceback
            error_msg = f"验证过程出错: {str(e)}"
            stack_trace = traceback.format_exc()
            print(f"{error_msg}\n{stack_trace}")
            self.progress_update.emit(error_msg)
            self.error_occurred.emit(error_msg)  # 发送错误信号

            # 发送空的验证结果以避免UI阻塞
            empty_results = {alg_name: {"valid": True, "confidence": 0.0, "relative_confidence": 0.0,
                                        "error": str(e)}
                             for alg_name in self.algorithm_results.keys()}
            self.validation_complete.emit(empty_results)
