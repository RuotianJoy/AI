from flask import Flask, render_template, request, jsonify, send_file
import json
from datetime import datetime
import os
import sys
import time
import itertools
import io
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle

# 添加父目录到系统路径以导入算法模块
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from genetic_algorithm import GeneticOptimizer
from simulated_annealing import SimulatedAnnealingOptimizer
from greedy_algorithm import r1, r2
from greedy_optimizer import GreedyOptimizer
from solution_validator import SolutionValidator

app = Flask(__name__)

# 确保数据目录存在
DATA_DIR = 'data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/compare')
def compare():
    """算法比较分析页面"""
    return render_template('compare.html')

@app.route('/api/calculate', methods=['POST'])
def calculate():
    try:
        data = request.json
        m = int(data['m'])
        n = int(data['n'])
        k = int(data['k'])
        j = int(data['j'])
        s = int(data['s'])
        f = int(data['f'])
        algorithm = data['algorithm']
        samples = data['samples']

        # 根据选择的算法创建优化器并运行优化
        if algorithm == 'greedy':
            # 使用贪心算法优化器
            start_time = time.time()
            
            try:
                # 创建贪心优化器并运行优化
                optimizer = GreedyOptimizer(samples, j, s, k, f)
                solution = optimizer.optimize()
                
                execution_time = time.time() - start_time
                
                # 检查结果是否为空列表（无可行解）
                if not solution:
                    return jsonify({
                        'success': False,
                        'message': '无可行解'
                    })
                
            except Exception as e:
                print(f"Greedy algorithm error: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': f'贪心算法计算错误: {str(e)}'
                })
                
        elif algorithm == 'genetic':
            optimizer = GeneticOptimizer(samples, j, s, k, f)
            # 记录开始时间
            start_time = time.time()
            solution = optimizer.optimize(population_size=100, generations=150)
            # 计算执行时间
            execution_time = time.time() - start_time
        else:
            optimizer = SimulatedAnnealingOptimizer(samples, j, s, k, f)
            # 记录开始时间
            start_time = time.time()
            solution = optimizer.optimize(
                initial_temp=10.0,
                cooling_rate=0.995,
                stopping_temp=0.0001,
                max_iterations=100000
            )
            # 计算执行时间
            execution_time = time.time() - start_time

        # 格式化结果
        result = {
            'success': True,
            'result': solution,
            'message': f'Calculation completed in {execution_time:.2f} seconds'
        }

        # 保存结果到文件
        save_result(data, result)

        return jsonify(result)

    except Exception as e:
        print(f"Error in calculate: {str(e)}")  # 添加错误日志
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/compare_calculate', methods=['POST'])
def compare_calculate():
    """同时运行指定的算法，并提供评估指标"""
    try:
        data = request.json
        m = int(data['m'])
        n = int(data['n'])
        k = int(data['k'])
        j = int(data['j'])
        s = int(data['s'])
        f = int(data['f'])
        algorithm = data['algorithm']
        samples = data['samples']

        start_time = time.time()
        solution = []
        metrics = {}
        
        # 根据指定的算法运行优化
        if algorithm == 'greedy':
            # 使用贪心算法优化器
            try:
                optimizer = GreedyOptimizer(samples, j, s, k, f)
                solution = optimizer.optimize()
                
                # 如果解为空，返回错误
                if not solution:
                    return jsonify({
                        'success': False,
                        'message': '贪心算法无法找到可行解'
                    })
                
                # 计算评估指标
                metrics = calculate_metrics(solution, samples, j, s, k, f)
            except Exception as e:
                print(f"Greedy algorithm error: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': f'贪心算法计算错误: {str(e)}'
                })
        
        elif algorithm == 'genetic':
            try:
                optimizer = GeneticOptimizer(samples, j, s, k, f)
                solution = optimizer.optimize(population_size=100, generations=150)
                
                # 计算评估指标
                metrics = calculate_metrics(solution, samples, j, s, k, f)
            except Exception as e:
                print(f"Genetic algorithm error: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': f'遗传算法计算错误: {str(e)}'
                })
        
        elif algorithm == 'annealing':
            try:
                optimizer = SimulatedAnnealingOptimizer(samples, j, s, k, f)
                solution = optimizer.optimize(
                    initial_temp=10.0,
                    cooling_rate=0.995,
                    stopping_temp=0.0001,
                    max_iterations=100000
                )
                
                # 计算评估指标
                metrics = calculate_metrics(solution, samples, j, s, k, f)
            except Exception as e:
                print(f"Simulated annealing error: {str(e)}")
                return jsonify({
                    'success': False,
                    'message': f'模拟退火算法计算错误: {str(e)}'
                })
        
        else:
            return jsonify({
                'success': False,
                'message': f'未知算法: {algorithm}'
            })

        # 计算执行时间
        execution_time = time.time() - start_time

        # 格式化结果
        result = {
            'success': True,
            'result': solution,
            'metrics': metrics,
            'message': f'计算完成，用时 {execution_time:.2f} 秒'
        }

        return jsonify(result)

    except Exception as e:
        print(f"Error in compare_calculate: {str(e)}")
        return jsonify({
            'success': False,
            'message': str(e)
        })

def calculate_metrics(solution, samples, j, s, k, f):
    """计算解决方案的评估指标，使用SolutionValidator"""
    try:
        # 使用SolutionValidator来验证解决方案并计算置信度
        validator = SolutionValidator(samples, j, s, k, f)
        is_valid, confidence, details = validator.validate_solution(solution)
        
        # 调试日志
        print(f"SolutionValidator结果: valid={is_valid}, confidence={confidence}")
        
        # 如果高级指标可用，则进行基准测试
        if hasattr(validator, 'benchmark_solution'):
            benchmark = validator.benchmark_solution(solution)
            confidence = benchmark.get('confidence', confidence)
            # 调试日志
            print(f"Benchmark置信度: {confidence}")
            # 从高级指标中提取其他数据
            advanced_metrics = benchmark.get('advanced_metrics', {})
        else:
            # 使用基本评估数据
            advanced_metrics = details.get('metrics', {}) if isinstance(details, dict) else {}
        
        # 确保最小置信度不为零（除非解决方案完全无效）
        if confidence < 0.01 and is_valid:
            confidence = 0.01
            print("置信度过低，设置为最小值0.01")
        
        # 1. 计算覆盖率（从验证器的详细信息中提取）
        if isinstance(details, dict) and 'coverage_ratio' in details:
            coverage = details['coverage_ratio'] * 100
        else:
            # 手动计算覆盖率
            j_subsets = list(itertools.combinations(samples, j))
            covered_count = 0
            
            for j_sub in j_subsets:
                j_set = set(j_sub)
                cover_count = sum(1 for group in solution if len(set(group) & j_set) >= s)
                if cover_count >= f:
                    covered_count += 1
            
            coverage = (covered_count / len(j_subsets)) * 100 if j_subsets else 100
        
        # 2. 计算平均组大小
        avg_group_size = sum(len(group) for group in solution) / len(solution) if solution else 0
        
        # 3. 计算样本利用率（使用过的样本 / 总样本数）
        used_samples = set()
        for group in solution:
            for sample in group:
                used_samples.add(sample)
        
        utilization = len(used_samples) / len(samples) if samples else 0
        
        # 构建返回值并记录日志
        result = {
            'coverage': coverage,
            'avgGroupSize': avg_group_size,
            'utilization': utilization,
            'confidence': confidence * 100  # 转换为百分比
        }
        print(f"最终计算的指标: {result}")
        return result
    except Exception as e:
        print(f"Error calculating metrics with SolutionValidator: {str(e)}")
        # 如果SolutionValidator出错，回退到原始计算方法
        return calculate_metrics_fallback(solution, samples, j, s, k, f)

def calculate_metrics_fallback(solution, samples, j, s, k, f):
    """原始的评估指标计算方法（作为备用）"""
    try:
        # 计算组的总体积
        total_size = sum(len(group) for group in solution)
        
        # 计算平均组大小
        avg_group_size = total_size / len(solution) if solution else 0
        
        # 计算样本利用率（使用过的样本 / 总样本数）
        used_samples = set()
        for group in solution:
            for sample in group:
                used_samples.add(sample)
        utilization = len(used_samples) / len(samples) if samples else 0
        
        # 基本置信度（根据组大小和组数量）
        confidence = min(avg_group_size / 10, 0.5) + min(utilization, 0.5)
        confidence = min(max(confidence, 0.01), 1.0)  # 限制在0.01到1.0之间
        
        # 覆盖率计算
        if j <= len(samples):
            j_subsets = list(itertools.combinations(samples, j))
            covered_count = 0
            
            for j_sub in j_subsets:
                j_set = set(j_sub)
                cover_count = sum(1 for group in solution if len(set(group) & j_set) >= s)
                if cover_count >= f:
                    covered_count += 1
            
            coverage = (covered_count / len(j_subsets)) * 100 if j_subsets else 100
        else:
            coverage = 0  # j大于样本数量，无法形成j子集
        
        return {
            "coverage": coverage,
            "avgGroupSize": avg_group_size,
            "utilization": utilization * 100,  # 转换为百分比
            "confidence": confidence * 100  # 转换为百分比
        }
    except Exception as e:
        print(f"Error in fallback metrics calculation: {str(e)}")
        return {
            'coverage': 0,
            'avgGroupSize': 0,
            'utilization': 0,
            'confidence': 0
        }

@app.route('/api/export_pdf', methods=['POST'])
def export_pdf():
    """Generate and export a PDF report of algorithm comparison results"""
    try:
        data = request.json
        
        # Extract data from request
        parameters = data.get('parameters', {})
        samples = data.get('samples', [])
        results = data.get('results', {})
        conclusion = data.get('conclusion', 'No conclusion available')
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=letter)
        elements = []
        
        # Get styles
        styles = getSampleStyleSheet()
        title_style = styles['Title']
        heading_style = styles['Heading1']
        subheading_style = styles['Heading2']
        normal_style = styles['Normal']
        
        # Create custom styles
        section_style = ParagraphStyle(
            'SectionTitle',
            parent=styles['Heading2'],
            spaceBefore=12,
            spaceAfter=6
        )
        
        code_style = ParagraphStyle(
            'CodeStyle',
            parent=styles['Normal'],
            fontName='Courier',
            fontSize=9,
            spaceBefore=0,
            spaceAfter=0
        )
        
        # Title
        elements.append(Paragraph("Algorithm Comparison Report", title_style))
        elements.append(Spacer(1, 12))
        
        # Time stamp
        date_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        elements.append(Paragraph(f"Generated: {date_str}", normal_style))
        elements.append(Spacer(1, 24))
        
        # Parameters section
        elements.append(Paragraph("Parameters", heading_style))
        
        param_data = [
            ["Parameter", "Value"],
            ["Total Samples (m)", str(parameters.get('m', '-'))],
            ["Selected Samples (n)", str(parameters.get('n', '-'))],
            ["Combination Size (k)", str(parameters.get('k', '-'))],
            ["Subset Parameter (j)", str(parameters.get('j', '-'))],
            ["Coverage Parameter (s)", str(parameters.get('s', '-'))],
            ["Coverage Times (f)", str(parameters.get('f', '-'))]
        ]
        
        param_table = Table(param_data, colWidths=[200, 200])
        param_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(param_table)
        elements.append(Spacer(1, 12))
        
        # Samples section
        elements.append(Paragraph("Input Samples", section_style))
        samples_text = ", ".join(samples)
        elements.append(Paragraph(samples_text, normal_style))
        elements.append(Spacer(1, 12))
        
        # Results section
        elements.append(Paragraph("Results Summary", heading_style))
        
        # Create comparison table
        comparison_data = [
            ["Algorithm", "Combinations", "Runtime (s)", "Coverage (%)", "Avg Group Size", "Utilization (%)", "Confidence (%)"]
        ]
        
        algorithms = {
            'greedy': 'Greedy Algorithm',
            'genetic': 'Genetic Algorithm',
            'annealing': 'Simulated Annealing'
        }
        
        for alg_key, alg_name in algorithms.items():
            if alg_key in results:
                result = results[alg_key]
                metrics = result.get('metrics', {})
                
                row = [
                    alg_name,
                    str(len(result.get('combinations', []))),
                    str(result.get('executionTime', '-')),
                    f"{metrics.get('coverage', '-'):.1f}" if metrics.get('coverage') is not None else '-',
                    f"{metrics.get('avgGroupSize', '-'):.2f}" if metrics.get('avgGroupSize') is not None else '-',
                    f"{metrics.get('utilization', '-'):.1f}" if metrics.get('utilization') is not None else '-',
                    f"{metrics.get('confidence', '-'):.1f}" if metrics.get('confidence') is not None else '-'
                ]
                comparison_data.append(row)
        
        comparison_table = Table(comparison_data, colWidths=[120, 80, 70, 70, 70, 70, 70])
        comparison_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.black),
            ('ALIGN', (0, 0), (-1, 0), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('ALIGN', (1, 1), (-1, -1), 'CENTER'),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ]))
        
        elements.append(comparison_table)
        elements.append(Spacer(1, 24))
        
        # Display combinations for each algorithm
        elements.append(Paragraph("Combination Details", heading_style))
        
        for alg_key, alg_name in algorithms.items():
            if alg_key in results:
                result = results[alg_key]
                elements.append(Paragraph(f"{alg_name}", subheading_style))
                
                combinations = result.get('combinations', [])
                if combinations:
                    # Show combinations
                    elements.append(Paragraph(f"Combinations found: {len(combinations)}", section_style))
                    
                    # Show all combinations
                    for i in range(len(combinations)):
                        combo = combinations[i]
                        combo_str = f"{i+1}. {', '.join(combo)}"
                        elements.append(Paragraph(combo_str, code_style))
                else:
                    elements.append(Paragraph("No combinations found", normal_style))
                
                elements.append(Spacer(1, 12))
        
        # Conclusion section
        elements.append(Paragraph("Analysis Conclusion", heading_style))
        elements.append(Paragraph(conclusion, normal_style))
        
        # Build PDF
        doc.build(elements)
        
        # Reset buffer position and return PDF
        buffer.seek(0)
        return send_file(
            buffer,
            mimetype='application/pdf',
            as_attachment=True,
            download_name=f'algorithm_comparison_{datetime.now().strftime("%Y%m%d_%H%M%S")}.pdf'
        )
        
    except Exception as e:
        print(f"Error generating PDF: {str(e)}")
        return jsonify({
            'success': False,
            'message': f'Error generating PDF: {str(e)}'
        }), 500

@app.route('/api/results', methods=['GET'])
def get_results():
    try:
        results = []
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith('.json'):
                    with open(os.path.join(DATA_DIR, filename), 'r') as f:
                        results.append(json.load(f))
        return jsonify(results)
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

@app.route('/api/results/<timestamp>', methods=['DELETE'])
def delete_result(timestamp):
    try:
        success = False
        message = "记录未找到"
        
        if os.path.exists(DATA_DIR):
            for filename in os.listdir(DATA_DIR):
                if filename.endswith('.json') and timestamp in filename:
                    file_path = os.path.join(DATA_DIR, filename)
                    os.remove(file_path)
                    success = True
                    message = "记录删除成功"
                    break
        
        return jsonify({
            'success': success,
            'message': message
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'message': str(e)
        })

def save_result(input_data, result):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'result_{timestamp}.json'
    
    data = {
        'timestamp': timestamp,
        'input': input_data,
        'result': result
    }
    
    with open(os.path.join(DATA_DIR, filename), 'w') as f:
        json.dump(data, f, indent=2)

@app.route('/history')
def history():
    return render_template('history.html')

if __name__ == '__main__':
    app.run(debug=True)