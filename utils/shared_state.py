"""Module for sharing state between components"""
import os
import sys
import threading
import time

# Flag to signal graceful termination of benchmarks
TERMINATE_REQUESTED = False
_trial_in_progress = False

def set_terminate():
    """Set the termination flag"""
    global TERMINATE_REQUESTED
    TERMINATE_REQUESTED = True
    print("\n\033[1;33mTermination flag set - will stop after current trial\033[0m")
    print("\033[1;33mPress 's' again to force immediate termination\033[0m")

def set_trial_status(status):
    """Update whether a trial is currently running"""
    global _trial_in_progress
    _trial_in_progress = status

def should_terminate():
    """Check if termination has been requested"""
    return TERMINATE_REQUESTED