"""Top-level entry point for NeuroPest standalone executable."""
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from neuropest.app import main
    main()
