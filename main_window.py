from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QGroupBox, QRadioButton,
                             QListWidget, QMessageBox, QTabWidget, QSpinBox, QFormLayout,
                             QTextEdit, QFileDialog, QGridLayout, QApplication, QSplitter,
                             QFrame, QScrollArea, QComboBox, QListWidgetItem, QStyleFactory, QSizePolicy,
                             QTextBrowser, QProgressBar)
from PyQt5.QtCore import Qt, QSettings, QThread, pyqtSignal, QSize, QMargins, QPropertyAnimation, QEasingCurve
from PyQt5.QtGui import QFont, QIntValidator, QColor, QPalette, QLinearGradient, QIcon, QPixmap, QFontDatabase
import os
import sqlite3
import random
import time
from genetic_algorithm import GeneticOptimizer
from simulated_annealing import SimulatedAnnealingOptimizer

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
            else:  # simulated_annealing
                optimizer = SimulatedAnnealingOptimizer(self.samples, self.j, self.s, self.k, self.f)
            
            # Setup progress callback for the optimizer
            optimizer.set_progress_callback(self.update_progress)
            
            # Run optimization
            best_solution = optimizer.optimize()
            
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

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("An Optimal Samples Selection System")
        self.setMinimumSize(1000, 700)
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
        subtitle_label = QLabel("Genetic Algorithm Optimization")
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
        
        # Set parameter input page
        self.setup_input_tab(input_tab)
        
        # Set result management page
        self.setup_results_tab(results_tab)
        
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
        m_label = QLabel("Total Sample Number (m):")
        m_label.setStyleSheet(label_style)
        m_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.m_input = QSpinBox()
        self.m_input.setRange(45, 54)
        self.m_input.setValue(45)
        param_layout.addWidget(m_label, 0, 0)
        param_layout.addWidget(self.m_input, 0, 1)
        
        # n parameter
        n_label = QLabel("Selected Sample Number (n):")
        n_label.setStyleSheet(label_style)
        n_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.n_input = QSpinBox()
        self.n_input.setRange(7, 25)
        self.n_input.setValue(7)
        param_layout.addWidget(n_label, 0, 2)
        param_layout.addWidget(self.n_input, 0, 3)
        
        # k parameter
        k_label = QLabel("Combination Size (k):")
        k_label.setStyleSheet(label_style)
        k_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.k_input = QSpinBox()
        self.k_input.setRange(4, 7)
        self.k_input.setValue(6)  # Default value is 6
        param_layout.addWidget(k_label, 0, 4)
        param_layout.addWidget(self.k_input, 0, 5)
        
        # Second row
        # j parameter
        j_label = QLabel("Subset Parameter (j):")
        j_label.setStyleSheet(label_style)
        j_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.j_input = QSpinBox()
        self.j_input.setRange(3, 25)
        self.j_input.setValue(4)
        param_layout.addWidget(j_label, 1, 0)
        param_layout.addWidget(self.j_input, 1, 1)
        
        # s parameter
        s_label = QLabel("Coverage Parameter (s):")
        s_label.setStyleSheet(label_style)
        s_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.s_input = QSpinBox()
        self.s_input.setRange(3, 7)
        self.s_input.setValue(4)
        param_layout.addWidget(s_label, 1, 2)
        param_layout.addWidget(self.s_input, 1, 3)
        
        # f parameter
        f_label = QLabel("Coverage Times (f):")
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
        
        # Connect manual/random selection buttons
        self.random_select.toggled.connect(lambda checked: self.sample_input.setEnabled(not checked))
        
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
    
    def update_n_max(self):
        """Update n maximum value to m"""
        m_value = self.m_input.value()
        self.n_input.setMaximum(min(25, m_value))
    
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
        
        # Disable calculate button
        self.calculate_btn.setEnabled(False)
        self.calculate_btn.setText("Processing...")
        
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
        
        # Restore calculate button
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Execute")
        
        # Hide progress bar
        self.progress_bar.setVisible(False)
    
    def update_progress(self, message):
        """Update progress information"""
        self.result_display.append(message)
    
    def handle_error(self, error_message):
        """Handle calculation error"""
        self.result_display.append(f"Calculation error: {error_message}")
    
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
                ''', (m, n, k, j, s, f, self.current_run_id, timestamp, sample_count, self.execution_time, run_count, run_format, algorithm))
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
        
        # Disable save button
        self.save_btn.setEnabled(False)
        
        # Reset calculation button
        self.calculate_btn.setEnabled(True)
        self.calculate_btn.setText("Execute") 