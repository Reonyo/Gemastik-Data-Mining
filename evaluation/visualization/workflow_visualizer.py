"""
Workflow Visualization Module

Generates visualizations and qualitative analysis artifacts
for the multi-agent system workflow traces.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import networkx as nx
from collections import defaultdict, Counter

try:
    import plotly.graph_objects as go
    import plotly.subplots as sp
    from plotly.offline import plot
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False
    print("Warning: Plotly not available. Some interactive visualizations will be skipped.")


class WorkflowVisualizer:
    """Generates visual analysis of multi-agent workflows."""
    
    def __init__(self, output_dir: str = "evaluation/results/visualizations"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Set style
        plt.style.use('seaborn-v0_8-whitegrid')
        sns.set_palette("husl")
    
    def analyze_workflow_traces(self, multi_agent_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze workflow traces to extract patterns and insights."""
        analysis = {
            "agent_usage": defaultdict(int),
            "transition_patterns": defaultdict(int),
            "iteration_distribution": [],
            "decision_points": [],
            "collaboration_matrix": defaultdict(lambda: defaultdict(int)),
            "performance_by_question_type": defaultdict(list),
            "workflow_efficiency": {}
        }
        
        for result in multi_agent_results:
            workflow_trace = result.get("workflow_trace", {})
            
            # Agent usage analysis
            agent_history = workflow_trace.get("agent_history", [])
            for agent in agent_history:
                analysis["agent_usage"][agent] += 1
            
            # Transition patterns
            for i in range(len(agent_history) - 1):
                transition = f"{agent_history[i]} → {agent_history[i+1]}"
                analysis["transition_patterns"][transition] += 1
            
            # Collaboration matrix
            for i, agent1 in enumerate(agent_history):
                for j, agent2 in enumerate(agent_history):
                    if i != j:
                        analysis["collaboration_matrix"][agent1][agent2] += 1
            
            # Iteration analysis
            iterations = workflow_trace.get("iteration_count", 1)
            analysis["iteration_distribution"].append(iterations)
            
            # Performance by type
            question_type = "case_study" if "case_id" in result else "multiple_choice"
            analysis["performance_by_question_type"][question_type].append({
                "iterations": iterations,
                "agents_used": len(set(agent_history)),
                "total_steps": len(agent_history)
            })
            
            # Decision points
            if "agent_debug_log" in result:
                debug_log = result["agent_debug_log"]
                for entry in debug_log:
                    if isinstance(entry, dict) and "decision" in entry:
                        analysis["decision_points"].append(entry)
        
        # Calculate efficiency metrics
        if analysis["iteration_distribution"]:
            analysis["workflow_efficiency"] = {
                "avg_iterations": np.mean(analysis["iteration_distribution"]),
                "median_iterations": np.median(analysis["iteration_distribution"]),
                "max_iterations": max(analysis["iteration_distribution"]),
                "min_iterations": min(analysis["iteration_distribution"])
            }
        
        return analysis
    
    def create_agent_usage_chart(self, analysis: Dict[str, Any]) -> str:
        """Create agent usage frequency chart."""
        agent_usage = dict(analysis["agent_usage"])
        
        if not agent_usage:
            return None
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        agents = list(agent_usage.keys())
        usage_counts = list(agent_usage.values())
        
        bars = ax.bar(agents, usage_counts, alpha=0.8)
        ax.set_title("Agent Usage Frequency", fontsize=14, fontweight='bold')
        ax.set_xlabel("Agent Type", fontsize=12)
        ax.set_ylabel("Usage Count", fontsize=12)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        chart_path = self.output_dir / "agent_usage_frequency.png"
        plt.savefig(chart_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(chart_path)
    
    def create_collaboration_matrix_heatmap(self, analysis: Dict[str, Any]) -> str:
        """Create agent collaboration matrix heatmap."""
        collab_matrix = analysis["collaboration_matrix"]
        
        if not collab_matrix:
            return None
        
        # Convert to DataFrame
        agents = list(set(list(collab_matrix.keys()) + 
                         [agent for subdict in collab_matrix.values() for agent in subdict.keys()]))
        
        matrix_data = []
        for agent1 in agents:
            row = []
            for agent2 in agents:
                count = collab_matrix[agent1].get(agent2, 0)
                row.append(count)
            matrix_data.append(row)
        
        df = pd.DataFrame(matrix_data, index=agents, columns=agents)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(df, annot=True, fmt='d', cmap='Blues', ax=ax)
        ax.set_title("Agent Collaboration Matrix", fontsize=14, fontweight='bold')
        ax.set_xlabel("To Agent", fontsize=12)
        ax.set_ylabel("From Agent", fontsize=12)
        
        plt.tight_layout()
        
        heatmap_path = self.output_dir / "collaboration_matrix.png"
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(heatmap_path)
    
    def create_transition_flow_diagram(self, analysis: Dict[str, Any]) -> str:
        """Create workflow transition flow diagram."""
        transitions = dict(analysis["transition_patterns"])
        
        if not transitions:
            return None
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add edges with weights
        for transition, count in transitions.items():
            if " → " in transition:
                source, target = transition.split(" → ")
                G.add_edge(source, target, weight=count)
        
        # Create visualization
        plt.figure(figsize=(12, 8))
        
        # Position nodes using spring layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Draw nodes
        node_sizes = [G.degree(node) * 500 + 1000 for node in G.nodes()]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, 
                              node_color='lightblue', alpha=0.7)
        
        # Draw edges with varying thickness based on weight
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        max_weight = max(weights) if weights else 1
        
        for (u, v), weight in zip(edges, weights):
            nx.draw_networkx_edges(G, pos, [(u, v)], 
                                  width=weight/max_weight * 5,
                                  alpha=0.6, edge_color='gray',
                                  arrowsize=20, arrowstyle='->')
        
        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold')
        
        # Add edge labels
        edge_labels = {(u, v): str(G[u][v]['weight']) for u, v in G.edges()}
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8)
        
        plt.title("Agent Workflow Transitions", fontsize=14, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        
        flow_path = self.output_dir / "workflow_transitions.png"
        plt.savefig(flow_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(flow_path)
    
    def create_iteration_distribution_plot(self, analysis: Dict[str, Any]) -> str:
        """Create iteration count distribution plot."""
        iterations = analysis["iteration_distribution"]
        
        if not iterations:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Histogram
        ax1.hist(iterations, bins=range(1, max(iterations) + 2), alpha=0.7, edgecolor='black')
        ax1.set_title("Iteration Count Distribution", fontsize=12, fontweight='bold')
        ax1.set_xlabel("Number of Iterations")
        ax1.set_ylabel("Frequency")
        ax1.grid(True, alpha=0.3)
        
        # Box plot
        ax2.boxplot(iterations, vert=True)
        ax2.set_title("Iteration Count Statistics", fontsize=12, fontweight='bold')
        ax2.set_ylabel("Number of Iterations")
        ax2.grid(True, alpha=0.3)
        
        # Add statistics text
        stats_text = f"""Mean: {np.mean(iterations):.1f}
Median: {np.median(iterations):.1f}
Std: {np.std(iterations):.1f}
Range: {min(iterations)}-{max(iterations)}"""
        
        ax2.text(1.1, np.mean(iterations), stats_text, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
        
        plt.tight_layout()
        
        iter_path = self.output_dir / "iteration_distribution.png"
        plt.savefig(iter_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(iter_path)
    
    def create_performance_comparison_chart(self, analysis: Dict[str, Any]) -> str:
        """Create performance comparison by question type."""
        perf_data = analysis["performance_by_question_type"]
        
        if not perf_data:
            return None
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        question_types = list(perf_data.keys())
        colors = ['skyblue', 'lightcoral']
        
        # Iterations by question type
        ax1 = axes[0, 0]
        iterations_data = [
            [item["iterations"] for item in perf_data[qtype]]
            for qtype in question_types
        ]
        
        bp1 = ax1.boxplot(iterations_data, labels=question_types, patch_artist=True)
        for patch, color in zip(bp1['boxes'], colors):
            patch.set_facecolor(color)
        
        ax1.set_title("Iterations by Question Type")
        ax1.set_ylabel("Number of Iterations")
        
        # Agents used by question type
        ax2 = axes[0, 1]
        agents_data = [
            [item["agents_used"] for item in perf_data[qtype]]
            for qtype in question_types
        ]
        
        bp2 = ax2.boxplot(agents_data, labels=question_types, patch_artist=True)
        for patch, color in zip(bp2['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_title("Unique Agents Used by Question Type")
        ax2.set_ylabel("Number of Unique Agents")
        
        # Total steps by question type
        ax3 = axes[1, 0]
        steps_data = [
            [item["total_steps"] for item in perf_data[qtype]]
            for qtype in question_types
        ]
        
        bp3 = ax3.boxplot(steps_data, labels=question_types, patch_artist=True)
        for patch, color in zip(bp3['boxes'], colors):
            patch.set_facecolor(color)
        
        ax3.set_title("Total Workflow Steps by Question Type")
        ax3.set_ylabel("Number of Steps")
        
        # Summary statistics table
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        summary_data = []
        for qtype in question_types:
            data = perf_data[qtype]
            avg_iter = np.mean([item["iterations"] for item in data])
            avg_agents = np.mean([item["agents_used"] for item in data])
            avg_steps = np.mean([item["total_steps"] for item in data])
            
            summary_data.append([qtype, f"{avg_iter:.1f}", f"{avg_agents:.1f}", f"{avg_steps:.1f}"])
        
        table = ax4.table(cellText=summary_data,
                         colLabels=["Question Type", "Avg Iterations", "Avg Agents", "Avg Steps"],
                         cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title("Summary Statistics")
        
        plt.tight_layout()
        
        perf_path = self.output_dir / "performance_by_question_type.png"
        plt.savefig(perf_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(perf_path)
    
    def create_interactive_workflow_timeline(self, multi_agent_results: List[Dict[str, Any]]) -> str:
        """Create interactive timeline visualization using Plotly."""
        if not PLOTLY_AVAILABLE:
            return None
        
        # Prepare timeline data
        timeline_data = []
        
        for i, result in enumerate(multi_agent_results):
            workflow_trace = result.get("workflow_trace", {})
            agent_history = workflow_trace.get("agent_history", [])
            
            question_id = result.get("question_id", result.get("case_id", f"Question_{i+1}"))
            
            for step, agent in enumerate(agent_history):
                timeline_data.append({
                    "Question": question_id,
                    "Step": step + 1,
                    "Agent": agent,
                    "QuestionIndex": i
                })
        
        if not timeline_data:
            return None
        
        df = pd.DataFrame(timeline_data)
        
        # Create interactive timeline
        fig = go.Figure()
        
        unique_agents = df["Agent"].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(unique_agents)))
        
        for j, agent in enumerate(unique_agents):
            agent_data = df[df["Agent"] == agent]
            
            fig.add_trace(go.Scatter(
                x=agent_data["Step"],
                y=agent_data["QuestionIndex"],
                mode='markers+lines',
                name=agent,
                text=agent_data["Question"],
                hovertemplate=f"<b>{agent}</b><br>" +
                             "Question: %{text}<br>" +
                             "Step: %{x}<br>" +
                             "<extra></extra>",
                marker=dict(size=10, color=f"rgba({colors[j][0]*255},{colors[j][1]*255},{colors[j][2]*255},0.8)")
            ))
        
        fig.update_layout(
            title="Multi-Agent Workflow Timeline",
            xaxis_title="Workflow Step",
            yaxis_title="Question Index",
            hovermode='closest',
            width=1000,
            height=600
        )
        
        timeline_path = self.output_dir / "interactive_workflow_timeline.html"
        plot(fig, filename=str(timeline_path), auto_open=False)
        
        return str(timeline_path)
    
    def generate_comprehensive_report(self, multi_agent_results: List[Dict[str, Any]]) -> Dict[str, str]:
        """Generate comprehensive visual analysis report."""
        print("🎨 Generating workflow visualizations...")
        
        # Analyze workflow traces
        analysis = self.analyze_workflow_traces(multi_agent_results)
        
        # Generate all visualizations
        visualization_paths = {}
        
        print("  📊 Creating agent usage chart...")
        usage_chart = self.create_agent_usage_chart(analysis)
        if usage_chart:
            visualization_paths["agent_usage"] = usage_chart
        
        print("  🔥 Creating collaboration heatmap...")
        heatmap = self.create_collaboration_matrix_heatmap(analysis)
        if heatmap:
            visualization_paths["collaboration_matrix"] = heatmap
        
        print("  🌊 Creating workflow transitions...")
        flow_diagram = self.create_transition_flow_diagram(analysis)
        if flow_diagram:
            visualization_paths["workflow_transitions"] = flow_diagram
        
        print("  📈 Creating iteration distribution...")
        iter_plot = self.create_iteration_distribution_plot(analysis)
        if iter_plot:
            visualization_paths["iteration_distribution"] = iter_plot
        
        print("  ⚡ Creating performance comparison...")
        perf_chart = self.create_performance_comparison_chart(analysis)
        if perf_chart:
            visualization_paths["performance_comparison"] = perf_chart
        
        print("  🕒 Creating interactive timeline...")
        timeline = self.create_interactive_workflow_timeline(multi_agent_results)
        if timeline:
            visualization_paths["interactive_timeline"] = timeline
        
        # Save analysis data
        analysis_path = self.output_dir / "workflow_analysis_data.json"
        with open(analysis_path, 'w', encoding='utf-8') as f:
            # Convert defaultdicts to regular dicts for JSON serialization
            analysis_serializable = {
                "agent_usage": dict(analysis["agent_usage"]),
                "transition_patterns": dict(analysis["transition_patterns"]),
                "iteration_distribution": analysis["iteration_distribution"],
                "workflow_efficiency": analysis["workflow_efficiency"],
                "performance_by_question_type": {
                    k: v for k, v in analysis["performance_by_question_type"].items()
                }
            }
            json.dump(analysis_serializable, f, indent=2, ensure_ascii=False)
        
        visualization_paths["analysis_data"] = str(analysis_path)
        
        print(f"✅ Generated {len(visualization_paths)} visualizations in {self.output_dir}")
        
        return visualization_paths


def main():
    """Test visualization generation with sample data."""
    # Create sample data for testing
    sample_results = [
        {
            "question_id": "mc_1",
            "workflow_trace": {
                "agent_history": ["Senior Lawyer", "Legal Researcher", "Legal Editor"],
                "iteration_count": 2
            },
            "predicted_answer": "B"
        },
        {
            "case_id": "case_1",
            "workflow_trace": {
                "agent_history": ["Senior Lawyer", "Legal Researcher", "Case Study Analyst", "Legal Editor"],
                "iteration_count": 3
            },
            "agent_debug_log": [
                {"decision": "analyze_case", "agent": "Senior Lawyer"},
                {"decision": "research_law", "agent": "Legal Researcher"}
            ]
        }
    ]
    
    visualizer = WorkflowVisualizer()
    paths = visualizer.generate_comprehensive_report(sample_results)
    
    print("\nGenerated visualizations:")
    for name, path in paths.items():
        print(f"  {name}: {path}")


if __name__ == "__main__":
    main()
