"""
Email Sender for Live Research Paper Collector
Handles actual email sending functionality
"""    

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import os
from typing import List, Optional
import json
from datetime import datetime

class EmailSender:
    def __init__(self):
        # Gmail SMTP configuration (you can change this)
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 587
        self.sender_email = None
        self.sender_password = None
        
        # Load email configuration from environment or file
        self._load_email_config()
    
    def _load_email_config(self):
        """Load email configuration from environment or config file"""
        # Try environment variables first
        self.sender_email = os.environ.get("EMAIL_SENDER")
        self.sender_password = os.environ.get("EMAIL_PASSWORD")
        
        # If not in environment, try to load from .env file
        if not self.sender_email:
            try:
                with open('.env', 'r') as f:
                    for line in f:
                        if line.startswith('EMAIL_SENDER='):
                            self.sender_email = line.split('=', 1)[1].strip()
                        elif line.startswith('EMAIL_PASSWORD='):
                            self.sender_password = line.split('=', 1)[1].strip()
            except:
                pass
        
        # If still not configured, set to demo mode
        if not self.sender_email or not self.sender_password:
            print("⚠️  Email not configured - using demo mode")
            self.demo_mode = True
        else:
            self.demo_mode = False
            print(f"✅ Email configured for: {self.sender_email}")
    
    def send_research_papers_email(
        self, 
        recipient_email: str, 
        subject: str, 
        papers: List[dict], 
        summaries: List[str],
        query: str
    ) -> dict:
        """
        Send research papers via email
        
        Args:
            recipient_email: Recipient email address
            subject: Email subject
            papers: List of paper dictionaries
            summaries: List of AI summaries
            query: Original research query
            
        Returns:
            Dictionary with success status and message
        """
        try:
            # Create email content
            email_content = self._format_email_content(query, papers, summaries)
            
            if self.demo_mode:
                # Demo mode - save to file instead of sending
                return self._demo_send_email(recipient_email, subject, email_content)
            
            # Real email sending
            return self._send_real_email(recipient_email, subject, email_content)
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Email sending failed: {str(e)}",
                "error": str(e)
            }
    
    def _format_email_content(self, query: str, papers: List[dict], summaries: List[str]) -> str:
        """Format the email content with research papers"""
        current_date = datetime.now().strftime("%B %d, %Y")
        
        email_content = f"""Subject: Latest Research Papers on {query} - {current_date}

Dear Researcher,

I hope this email finds you well. I've collected the latest research papers on "{query}" from arXiv.org for your review.

=== RESEARCH PAPERS COLLECTION ===

"""
        
        for i, (paper, summary) in enumerate(zip(papers, summaries), 1):
            authors = ', '.join(paper['authors'][:3])
            if len(paper['authors']) > 3:
                authors += '...'
            
            categories = ', '.join(paper['categories'][:3])
            
            email_content += f"""
{i}. {paper['title']}

Authors: {authors}
Published: {paper['published_date']}
Categories: {categories}

Summary: {summary}

Link: {paper['arxiv_url']}
PDF: {paper['pdf_url']}

---
"""
        
        email_content += f"""
=== COLLECTION SUMMARY ===
Total Papers: {len(papers)}
Collection Date: {current_date}
Source: arXiv.org
Query: {query}

=== NEXT STEPS ===
• Review the abstracts and summaries above
• Download papers of interest using the provided PDF links
• Contact me if you need additional papers or have questions

Best regards,
AI Research Assistant
Powered by Google Gemini & Playwright Automation

---
This email was automatically generated using AI-powered research collection.
For more information or to request additional research topics, please reply to this message.
"""
        
        return email_content
    
    def _demo_send_email(self, recipient_email: str, subject: str, content: str) -> dict:
        """Demo mode - save email to file"""
        try:
            # Create email filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"email_demo_{timestamp}.txt"
            
            # Save email content to file
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(f"To: {recipient_email}\n")
                f.write(f"From: {self.sender_email or 'demo@example.com'}\n")
                f.write(f"Subject: {subject}\n")
                f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 50 + "\n")
                f.write(content)
            
            return {
                "success": True,
                "message": f"Demo mode: Email saved to {filename}",
                "demo_file": filename,
                "recipient": recipient_email,
                "subject": subject
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"Demo email save failed: {str(e)}",
                "error": str(e)
            }
    
    def _send_real_email(self, recipient_email: str, subject: str, content: str) -> dict:
        """Send real email using SMTP"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = recipient_email
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(content, 'plain'))
            
            # Create SMTP session
            context = ssl.create_default_context()
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls(context=context)
            
            # Login and send
            server.login(self.sender_email, self.sender_password)
            text = msg.as_string()
            server.sendmail(self.sender_email, recipient_email, text)
            server.quit()
            
            return {
                "success": True,
                "message": f"Email successfully sent to {recipient_email}",
                "recipient": recipient_email,
                "subject": subject,
                "sent_at": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "success": False,
                "message": f"SMTP sending failed: {str(e)}",
                "error": str(e)
            }
    
    def get_email_status(self) -> dict:
        """Get current email configuration status"""
        return {
            "configured": not self.demo_mode,
            "sender_email": self.sender_email if not self.demo_mode else None,
            "demo_mode": self.demo_mode,
            "smtp_server": self.smtp_server,
            "smtp_port": self.smtp_port
        }

# Usage example
if __name__ == "__main__":
    sender = EmailSender()
    status = sender.get_email_status()
    print("Email Status:", status)
    
    # Example papers data
    papers = [
        {
            "title": "Test Paper 1",
            "authors": ["Author 1", "Author 2"],
            "abstract": "This is a test abstract",
            "arxiv_url": "https://arxiv.org/abs/1234",
            "pdf_url": "https://arxiv.org/pdf/1234.pdf",
            "published_date": "2024-01-01",
            "categories": ["cs.AI", "cs.LG"]
        }
    ]
    
    summaries = ["This is a test summary"]
    
    # Test email sending
    result = sender.send_research_papers_email(
        recipient_email="test@example.com",
        subject="Test Research Papers",
        papers=papers,
        summaries=summaries,
        query="test query"
    )
    
    print("Email Result:", result)
