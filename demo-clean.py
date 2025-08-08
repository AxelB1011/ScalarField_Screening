# demo.py - Clean Demo Script

import asyncio
import sys
import os
from dotenv import load_dotenv

load_dotenv()

async def demo_system():
    """Clean demo of the SEC QA System"""
    from pipeline import SECQASystem
    
    # Check required environment variables
    required_vars = ["EDGAR_IDENTITY", "GEMINI_API_KEY", "COCKROACH_DATABASE_URL"]
    missing = [var for var in required_vars if not os.getenv(var)]
    
    if missing:
        print(f"❌ Missing required environment variables: {missing}")
        print("Set them like:")
        for var in missing:
            if var == "EDGAR_IDENTITY":
                print(f'export {var}="YourName your.email@domain.com"')
            else:
                print(f'export {var}="your_api_key_here"')
        sys.exit(1)
    
    # Initialize system
    system = SECQASystem()
    if not await system.initialize():
        print("❌ System initialization failed")
        return
    
    print("🚀 SEC QA System initialized successfully!")
    
    # Ingest sample companies
    print("\n📊 Ingesting sample companies...")
    companies = ["AAPL"]
    results = await system.ingest_companies(companies, limit_per_company=2)
    
    for ticker, result in results.items():
        if result['errors']:
            print(f" ⚠️ {ticker}: {len(result['errors'])} errors")
            for error in result['errors'][:3]:  # Show first 3 errors
                print(f"   - {error}")
        else:
            print(f" ✅ {ticker}: {result['chunks_processed']} chunks in {result['processing_time']:.1f}s")
    
    # Demo questions
    print("\n❓ Demo questions:")
    test_cases = [
        "What are Apple's main risk factors?",
        "What was Apple's revenue in the latest quarter?", 
        "How much does Apple spend on R&D?",
        "What are Apple's main business segments?"
    ]
    
    for question in test_cases:
        print(f"\n🤔 {question}")
        try:
            result = await system.ask_question(question, max_sources=5)
            
            print(f" 📈 Confidence: {result.confidence:.2f}")
            print(f" 🎯 Routing: {result.routing_info['reasoning']}")
            print(f" 📚 Sources: ({len(result.sources)}) sources: {[[i+1, r['document_url']] for i, r in enumerate(result.sources)]}")
            print(f" 🤖 Provider: {result.provider_used}")
            print(f" ⏱️ Time: {result.processing_time:.1f}s")
            print(f" 💬 Answer: {result.answer[:300]}...")
            
            if result.sources:
                print("   📋 Top sources:")
                for source in result.sources[:3]:
                    print(f"     - {source['ticker']} {source['form_type']} - {source['section_title'][:50]}...")
            
        except Exception as e:
            print(f" ❌ Error: {e}")
    
    # Show system status
    print("\n📊 System Status:")
    status = await system.get_status()
    for key, value in status.items():
        if key != "features":
            print(f" {key}: {value}")
    
    print("\n✨ Features:")
    for feature in status.get("features", []):
        print(f" {feature}")
    
    print("\n✅ Demo complete!")

def main():
    """Main entry point"""
    asyncio.run(demo_system())

if __name__ == "__main__":
    main()