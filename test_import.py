try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
    print("Import successful")
except ImportError as e:
    print(f"Import failed: {e}")
    # Try alternative imports
    try:
        from transformers import AutoModelForCausalLM as AutoModelForCausalLMAlt
        print("Alternative import worked")
    except:
        print("Alternative also failed")
