import logging
import colorlog
import os

def setup_logging():
    """Set up colored logging with file handler."""
    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(log_color)s%(levelname)s: %(message)s',
        log_colors={
            'DEBUG': 'cyan', 'INFO': 'green', 'WARNING': 'yellow',
            'ERROR': 'red', 'CRITICAL': 'red,bg_white'
        }))
    logger = logging.getLogger('vaccine_analysis')
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

    # Add file handler to save logs
    file_handler = logging.FileHandler('summaries/analysis_log.txt')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    return logger

logger = setup_logging()

def save_summary_to_md(content, filename):
    """Save content to a markdown file."""
    with open(f'summaries/{filename}', 'w', encoding='utf-8') as f:
        f.write(content)
    logger.info(f"Saved summary to {filename}")