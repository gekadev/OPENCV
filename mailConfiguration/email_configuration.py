
'''
import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email_setting import *
# Define email sender and receiver




def sendingMail(from_email=email_sender ,to_email=email_receiver):
    # Create the email headers and subject
    sent_subject = "SmokerX  Alert"

    # Create the email body
    sent_body = 'This is an automatic Mail Sender by SmokerX App.n\
    There is a smoking person detected as attached below.'

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = email_sender
    message["To"] = email_receiver
    message["Subject"] = sent_subject

    # Add body to email
    message.attach(MIMEText(sent_body, "plain"))

    # Convert the message to a string
    email_text = message.as_string()

    try:
        # Connect to the SMTP server using SSL
        context = ssl.create_default_context()
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context)
        server.login(email_sender, email_password)
        
        # Send the email
        server.sendmail(email_sender, [email_sender, email_receiver], email_text)
        
        # Close the server connection
        server.quit()
        return True

        print('Email sent successfully!')
    except Exception as exception:
        print(f"Error: {exception}")
        return False
        
'''        
'''   
if __name__ == '__main__':
    sendingMail()
'''     


import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import cv2
import numpy as np
from mailConfiguration.email_setting import *
from datetime import datetime



def sendingMail(from_email=email_sender, to_email=email_receiver,  img=None,personName=None):
    # Get current time and date
    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")
    
    # Create the email headers and subject
    sent_subject = "Smoking Activity Detected - Immediate Attention Required"

    # Create the email body
    sent_body = (
        f"This is an automated notification from the SmokerX App. A person identified as {personName} has been detected smoking, "
        "as captured in the attached image. Below are the details:\n"
        f"Person Name: {personName}\n"
        f"Time of Detection: {current_time}\n"
        f"Date of Detection: {current_date}\n"
        "Location: Fourth Floor, Building Number 5\n"
        "Please review the attached image for further verification.\n\n"
        "Best regards,\n"
        "SmokerX Monitoring Team"
    )

    # Create a multipart message and set headers
    message = MIMEMultipart()
    message["From"] = from_email
    message["To"] = to_email
    message["Subject"] = sent_subject

    # Add body to email
    message.attach(MIMEText(sent_body, "plain"))

    if img is not None:
        # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)
        if ret:
            image_data = buffer.tobytes()

            # Create an attachment part
            image_attachment = MIMEBase("application", "octet-stream")
            image_attachment.set_payload(image_data)
            encoders.encode_base64(image_attachment)
            image_attachment.add_header(
                "Content-Disposition",
                "attachment; filename=smoking_detected.jpg",
            )

            # Attach the image to the email
            message.attach(image_attachment)

    # Convert the message to a string
    email_text = message.as_string()

    try:
        # Connect to the SMTP server using SSL
        context = ssl.create_default_context()
        server = smtplib.SMTP_SSL('smtp.gmail.com', 465, context=context)
        server.login(email_sender, email_password)

        # Send the email
        server.sendmail(email_sender, [email_sender, email_receiver], email_text)

        # Close the server connection
        server.quit()
        print('Email sent successfully!')
        return True

    except Exception as exception:
        print(f"Error: {exception}")
        return False
