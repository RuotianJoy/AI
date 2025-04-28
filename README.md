# Optimal Samples Selection System

An intelligent system for selecting optimal sample combinations using genetic algorithms and simulated annealing optimization.

## Overview

This system helps users find the optimal combination of samples that satisfy specific coverage requirements. It provides two optimization algorithms:
- Genetic Algorithm
- Simulated Annealing

The system features a user-friendly GUI interface and maintains a database of optimization results for future reference.

## Features

- **Dual Algorithm Support**
  - Genetic Algorithm optimization
  - Simulated Annealing optimization
  - Real-time progress tracking
  - Detailed result visualization

- **Parameter Configuration**
  - Total sample number (m): 45-54
  - Selected sample number (n): 7-25
  - Combination size (k): 4-7
  - Subset parameter (j): ≥3
  - Coverage parameter (s): 3-7
  - Coverage times (f): ≥1

- **Sample Selection Methods**
  - Random selection
  - Manual input

- **Result Management**
  - Save optimization results to database
  - View historical optimization records
  - Export results to text files
  - Delete unwanted records

## Requirements

- Python 3.6+
- PyQt5
- SQLite3

## Installation

1. Clone the repository:
```bash
git clone [repository-url]
cd [repository-name]
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Run the application:
```bash
python main.py
```

2. Configure parameters:
   - Set total sample number (m)
   - Set selected sample number (n)
   - Set combination size (k)
   - Set subset parameter (j)
   - Set coverage parameter (s)
   - Set coverage times (f)
   - Choose optimization algorithm

3. Select samples:
   - Choose random selection or manual input
   - If manual input, enter sample numbers separated by commas

4. Execute optimization:
   - Click "Execute" button
   - Monitor progress in real-time
   - View results in the display area

5. Manage results:
   - Save results to database
   - View historical records
   - Export results to text files
   - Delete unwanted records

## Database Structure

The system uses SQLite3 database with the following tables:

- **runs**: Stores optimization run information
  - Parameters (m, n, k, j, s, f)
  - Timestamp
  - Execution time
  - Algorithm used
  - Run count
  - Formatted ID

- **samples**: Stores selected samples for each run
  - Run ID
  - Sample number

- **results**: Stores optimization results
  - Run ID
  - Group ID
  - Sample number

## File Structure

```
.
├── main.py                 # Main application entry point
├── main_window.py          # GUI implementation
├── genetic_algorithm.py    # Genetic algorithm implementation
├── simulated_annealing.py  # Simulated annealing implementation
├── results.db              # SQLite database
└── requirements.txt        # Python dependencies
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

For any questions or suggestions, please contact the project maintainers. 

