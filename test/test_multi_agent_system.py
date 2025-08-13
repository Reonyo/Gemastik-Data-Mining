"""
Test Multi-Agent System REAL Implementation

Tests the 5-agent legal analysis system with small dataset for debugging.
Use this for testing, NOT the main evaluation.
"""

import sys
import os
import asyncio
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from workflow.legal_workflow import create_legal_workflow

async def test_multi_agent_simple():
    """Test Multi-Agent system with simple legal question."""
    
    print("🧪 Testing Multi-Agent System - REAL IMPLEMENTATION")
    print("="*60)
    
    try:
        # Initialize the real multi-agent workflow
        workflow = create_legal_workflow()
        print("✅ Multi-Agent workflow initialized")
        
        # Test question
        test_question = """
        Saya ingin mendirikan CV (Commanditaire Vennootschap) di Indonesia. 
        Apa saja persyaratan hukum yang harus dipenuhi?
        """
        
        print(f"📝 Test Question: {test_question.strip()}")
        print("\n🔄 Running Multi-Agent Analysis...")
        
        # Run with timeout and limited iterations
        result = workflow.run_workflow(test_question, max_iterations=2)
        
        print("\n📊 RESULTS:")
        print(f"✅ Success: {result['success']}")
        print(f"📄 Final Document Length: {len(result.get('final_document', ''))}")
        print(f"🔍 Structured Facts: {len(result.get('structured_facts', []))}")
        print(f"📚 Legal Analysis Steps: {len(result.get('legal_analysis', []))}")
        print(f"🔄 Iterations: {result.get('workflow_metadata', {}).get('iteration_count', 0)}")
        print(f"🤖 Agents Used: {result.get('workflow_metadata', {}).get('agent_history', [])}")
        print(f"❌ Errors: {len(result.get('errors', []))}")
        
        if result.get('final_document'):
            print(f"\n📋 Final Document Preview:")
            print("-" * 40)
            print(result['final_document'][:500] + "..." if len(result['final_document']) > 500 else result['final_document'])
        
        if result.get('errors'):
            print(f"\n⚠️ Errors Found:")
            for error in result['errors']:
                print(f"  - {error}")
        
        return result
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return None

async def test_multi_agent_mcq():
    """Test Multi-Agent system with MCQ question."""
    
    print("\n" + "="*60)
    print("🧪 Testing Multi-Agent System - MCQ Format")
    print("="*60)
    
    try:
        workflow = create_legal_workflow()
        
        mcq_question = """
        Dalam hukum Indonesia, apa yang dimaksud dengan asas legalitas?
        
        A. Setiap orang berhak mendapat bantuan hukum
        B. Tidak ada kejahatan tanpa undang-undang yang mengaturnya
        C. Hukum berlaku surut
        D. Semua orang sama di hadapan hukum
        
        Berikan analisis hukum yang komprehensif untuk setiap opsi dan pilih jawaban yang benar.
        """
        
        print(f"📝 MCQ Question: {mcq_question.strip()[:100]}...")
        print("\n🔄 Running Multi-Agent MCQ Analysis...")
        
        result = workflow.run_workflow(mcq_question, max_iterations=2)
        
        print(f"\n📊 MCQ RESULTS:")
        print(f"✅ Success: {result['success']}")
        print(f"📄 Analysis Length: {len(result.get('final_document', ''))}")
        
        if result.get('final_document'):
            print(f"\n📋 MCQ Analysis Preview:")
            print("-" * 40)
            print(result['final_document'][:700] + "..." if len(result['final_document']) > 700 else result['final_document'])
        
        return result
        
    except Exception as e:
        print(f"❌ MCQ Test failed: {e}")
        return None

async def main():
    """Run all Multi-Agent tests."""
    
    print("🚀 MULTI-AGENT SYSTEM TESTING - REAL IMPLEMENTATION")
    print("="*70)
    print("Purpose: Test real 5-agent system before evaluation")
    print("Expected: Multiple API calls, real legal analysis")
    print("="*70)
    
    # Test 1: Simple legal question
    result1 = await test_multi_agent_simple()
    
    # Test 2: MCQ format
    result2 = await test_multi_agent_mcq()
    
    print("\n" + "="*70)
    print("🎯 MULTI-AGENT TESTING SUMMARY")
    print("="*70)
    print(f"✅ Simple Legal Question: {'PASSED' if result1 and result1['success'] else 'FAILED'}")
    print(f"✅ MCQ Format Question: {'PASSED' if result2 and result2['success'] else 'FAILED'}")
    
    if result1 and result1['success'] and result2 and result2['success']:
        print("\n🎉 Multi-Agent System is READY for evaluation!")
    else:
        print("\n⚠️ Multi-Agent System needs fixes before evaluation")

if __name__ == "__main__":
    asyncio.run(main())
