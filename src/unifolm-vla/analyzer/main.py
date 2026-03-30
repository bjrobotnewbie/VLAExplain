# 🧠 VLAExplain — Interpreting Vision-Language-Action (VLA) Models
# Copyright (C) 2026 Yafei Shi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# For more information, please visit: https://github.com/bjrobotnewbie/VLAExplain

import os
import argparse
import logging

logger = logging.getLogger(__name__)
def setup_and_run():
    """Setup environment and run application in correct order"""
    logger.info(f"VLA_DATA_DIR configured: {os.getenv('VLA_DATA_DIR')}")
    
    # Import modules after environment setup
    from ui.interface import create_unified_interface
    from config.settings import Settings
    
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="VLA Model Attention Analysis Platform",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--host", type=str, default=Settings.SERVER_HOST, help="Server host address")
    parser.add_argument("--port", type=int, default=Settings.SERVER_PORT, help="Server port number")
    parser.add_argument("--inbrowser", action="store_true", help="Automatically open browser after startup")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    parser.add_argument("--n_action_steps", type=int, default=8, help="Num of action steps")
    
    args = parser.parse_args()
    
    # Validate configuration
    try:
        Settings.validate_paths()
    except FileNotFoundError as e:
        print(f"Configuration Error: {e}")
        return
    
    # Launch application
    try:
        vis = create_unified_interface(n_action_steps=args.n_action_steps)
        vis.queue()
        vis.launch(
            server_name=args.host,
            server_port=args.port,
            inbrowser=args.inbrowser,
            debug=args.debug,
        )

    except Exception as e:
        print(f"Application failed to start: {e}")
        raise


if __name__ == "__main__":
    setup_and_run()