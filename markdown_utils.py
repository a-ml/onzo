import re
from datetime import datetime

def format_section_header(role):
    """
    Format the section header for a role with improved markdown structure.
    
    Args:
        role (str): The role of the speaker (e.g., 'Moderator', 'Socialist', 'Capitalist')
    
    Returns:
        str: A formatted markdown section header
    """
    # Use different header levels for different roles
    header_mapping = {
        'Moderator': '##',   # Second-level header
        'Socialist': '###',  # Third-level header
        'Capitalist': '###', # Third-level header
        'System': '*',       # Italic for system messages
    }
    
    header_level = header_mapping.get(role, '###')
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    return f"{header_level} {role} - {timestamp}\n\n"

def format_message(message):
    """
    Format the message with markdown blockquote and some additional styling.
    
    Args:
        message (str): The message to be formatted
    
    Returns:
        str: A formatted markdown message
    """
    # Clean up the message: remove extra whitespaces, handle newlines
    cleaned_message = re.sub(r'\s+', ' ', message).strip()
    
    # Add syntax highlighting for code-like content
    if any(keyword in cleaned_message.lower() for keyword in ['code', 'function', 'method', 'class']):
        return f"> **Note:** ```\n> {cleaned_message}\n> ```\n\n"
    
    return f"> *{cleaned_message}*\n\n"

def format_conclusion(conclusion):
    """
    Format the conclusion with a distinct markdown structure.
    
    Args:
        conclusion (str): The concluding message of the debate
    
    Returns:
        str: A formatted markdown conclusion
    """
    # Clean up the conclusion
    cleaned_conclusion = re.sub(r'\s+', ' ', conclusion).strip()
    
    return (
        "## 🏁 Debate Conclusion\n\n"
        f"> **Final Insights:** *{cleaned_conclusion}*\n\n"
        "### Key Takeaways\n"
        "- Balanced perspectives were explored\n"
        "- Complex economic considerations were discussed\n"
    )

def generate_debate_metadata(topic, participants=None):
    """
    Generate metadata for the debate markdown document.
    
    Args:
        topic (str): The debate topic
        participants (list, optional): List of debate participants
    
    Returns:
        str: A markdown-formatted metadata section
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    participants = participants or ['Moderator', 'Socialist Perspective', 'Capitalist Perspective']
    
    metadata = (
        "# Economic Debate Transcript\n\n"
        f"**Date:** {timestamp}\n"
        f"**Topic:** {topic}\n\n"
        "## Participants\n"
    )
    
    for participant in participants:
        metadata += f"- {participant}\n"
    
    return metadata + "\n---\n\n"