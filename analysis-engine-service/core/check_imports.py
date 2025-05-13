"""
Simple script to check if modules can be imported.
"""

import sys
import os

# Add the parent directory to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.insert(0, "D:\\MD\\forex_trading_platform")

print(f"Python path: {sys.path}")

# Try to import the module
try:
    import analysis_engine
    print("Successfully imported analysis_engine")

    # Check if the confluence module exists
    try:
        import analysis_engine.analysis
        print("Successfully imported analysis_engine.analysis")

        # Check if the confluence module exists
        try:
            import analysis_engine.analysis.confluence
            print("Successfully imported analysis_engine.analysis.confluence")

            # Try to import the ConfluenceAnalyzer class
            try:
    """
    try class.
    
    Attributes:
        Add attributes here
    """

                from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
                print("Successfully imported ConfluenceAnalyzer")
                print("ConfluenceAnalyzer class exists and can be imported")
            except ImportError as e:
                print(f"Error importing ConfluenceAnalyzer: {e}")

                # Try with the underscore version
                try:
                    print("Trying with underscore in filename...")
                    from analysis_engine.analysis.confluence.confluence_analyzer import ConfluenceAnalyzer
                    print("Successfully imported ConfluenceAnalyzer with underscore")
                    print("ConfluenceAnalyzer class exists and can be imported")
                except ImportError as e:
                    print(f"Error importing ConfluenceAnalyzer with underscore: {e}")
        except ImportError as e:
            print(f"Error importing analysis_engine.analysis.confluence: {e}")
    except ImportError as e:
        print(f"Error importing analysis_engine.analysis: {e}")
except ImportError as e:
    print(f"Error importing analysis_engine: {e}")

# Try to import the RelatedPairsConfluenceAnalyzer class
try:
    """
    try class.
    
    Attributes:
        Add attributes here
    """

    from analysis_engine.multi_asset.related_pairs_confluence_detector import RelatedPairsConfluenceAnalyzer
    print("Successfully imported RelatedPairsConfluenceAnalyzer")
    print("RelatedPairsConfluenceAnalyzer class exists and can be imported")
except ImportError as e:
    print(f"Error importing RelatedPairsConfluenceAnalyzer: {e}")

print("Import check completed")
