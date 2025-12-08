try:
    import flash_attn
    print("Version:", flash_attn.__version__)
    print("Contents:", dir(flash_attn))
    from flash_attn import flash_attn_func, flash_attn_unpadded_func
    print("Found v1 functions")
except ImportError as e:
    print("ImportError:", e)
except Exception as e:
    print("Error:", e)
