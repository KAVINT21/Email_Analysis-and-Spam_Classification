import imaplib
import email
from email.header import decode_header
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import re
from datetime import datetime
from streamlit_tags import st_tags



import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt

def model_deployment():
    df = pd.read_csv("D:\\Data Mining - Waka\\spam.csv")
    df['spam'] = df['Category'].apply(lambda x: 1 if x == 'spam' else 0)
    X_train, X_test, y_train, y_test = train_test_split(df.Message, df.spam)
    v = CountVectorizer()
    X_train_count = v.fit_transform(X_train.values)
    model = MultinomialNB()
    model.fit(X_train_count, y_train)
    return model,v




def main():
    st.header(":mailbox: Email Analysis and Spam Classification")
    user = st.text_input("**Enter Email**:")
    password = st.text_input("**Enter Password**:", type='password')
    model, v = model_deployment()
    if st.button("Submit"):
       
        user = user
        password = password

        imap_url="imap.gmail.com"


        my_mail=imaplib.IMAP4_SSL(imap_url)

        my_mail.login(user,password)

        my_mail.select("Inbox")

        status, messages = my_mail.search(None, "ALL")
        message_ids = messages[0].split()

        body_list = []
        subject_list = []

        for msg_id in message_ids:
            # Fetch the email data
            status, msg_data = my_mail.fetch(msg_id, "(RFC822)")

            # Get the message part
            msg = email.message_from_bytes(msg_data[0][1])

            # Decode subject
            subject, encoding = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(encoding or "utf-8")

            # Get body
            if msg.is_multipart():
                for part in msg.walk():
                    if part.get_content_type() == "text/plain":
                        body = part.get_payload(decode=True).decode("utf-8")
                        body_list.append(body)
            else:
                body = msg.get_payload(decode=True).decode("utf-8")
                body_list.append(body)

            # Append subject to list
            subject_list.append(subject)

        timestamps = []
        senders = []

        # Fetch email data for each message ID
        for msg_id in message_ids:
            # Fetch INTERNALDATE
            status, internal_date_data = my_mail.fetch(msg_id, '(INTERNALDATE)')
            internal_date_match = re.search(r'INTERNALDATE "([^"]+)"', internal_date_data[0].decode('utf-8'))

            if internal_date_match:
                internal_date = internal_date_match.group(1)
                timestamp = datetime.strptime(internal_date, '%d-%b-%Y %H:%M:%S %z')
                timestamps.append(timestamp)

            # Fetch RFC822
            status, rfc822_data = my_mail.fetch(msg_id, '(RFC822)')
            msg = email.message_from_bytes(rfc822_data[0][1])

            # Get sender (FROM)
            sender = msg.get('From')
            senders.append(sender)

        # Search for all emails in the mailbox and get their message IDs
        message_ids = messages[0].split()

        # Convert message IDs from bytes to strings
        message_ids = [msg_id.decode("utf-8") for msg_id in message_ids]

        # Close the connection
        my_mail.close()
        my_mail.logout()




        cleaned_body_list = []

        # Iterate through each email content and clean up
        for content in body_list:
            # Remove '\r' and '\n'
            cleaned_content = re.sub(r'\r\n', ' ', content)

            # Remove &nbsp;
            cleaned_content = re.sub(r'&nbsp;', ' ', cleaned_content)

            # Remove image-related HTML tags
            cleaned_content = re.sub(r'<img[^>]+>', '', cleaned_content)

            # Remove other HTML tags
            cleaned_content = re.sub(r'<[^>]+>', '', cleaned_content)

            # Remove extra spaces
            cleaned_content = ' '.join(cleaned_content.split())

            # Append cleaned content to the list
            cleaned_body_list.append(cleaned_content)

        # Filter out empty messages from the cleaned_body_list
        cleaned_body_list = [content for content in cleaned_body_list if content.strip()]


        print(" -------------- Subjects ------------ ")
        emails_count = v.transform(subject_list)
        prediction=model.predict(emails_count)
        spams=[]
        for i in range(len(prediction)):
            if prediction[i]==1:
                print("##############################")
                print(subject_list[i])
                spams.append(subject_list[i])


        # Assuming 'cleaned_body_list' contains the cleaned email bodies
        vectorizer = CountVectorizer(stop_words='english')
        email_bodies_count = vectorizer.fit_transform(cleaned_body_list)
        feature_names = vectorizer.get_feature_names_out()

        # Get word frequencies
        word_frequencies = zip(feature_names, email_bodies_count.sum(axis=0).A1)

        # Sort words by frequency
        vectorizer = CountVectorizer(stop_words='english')
        email_bodies_count = vectorizer.fit_transform(cleaned_body_list)
        feature_names = vectorizer.get_feature_names_out()

        # Get word frequencies
        word_frequencies = zip(feature_names, email_bodies_count.sum(axis=0).A1)

        # Sort words by frequency
        sorted_words = sorted(word_frequencies, key=lambda x: x[1], reverse=True)[:5]

        # Create a Seaborn bar plot
        plt.figure(figsize=(3, 3))
        sns.barplot(x=[word[0] for word in sorted_words], y=[word[1] for word in sorted_words], palette='viridis')
        plt.xticks(rotation=90, ha='right',fontsize=7)
        plt.xlabel('Words',fontsize=7)
        plt.ylabel('Frequency',fontsize=7)
        plt.title('Top 10 Most Frequent Words in Email Bodies',fontsize=7)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Display the bar plot in Streamlit
        st.pyplot()

        sender_counts = pd.Series(senders).value_counts().reset_index(name='count')[:4]

        # Create a donut plot with Seaborn
        plt.figure(figsize=(8, 8))
        colors = sns.color_palette('pastel')[0:len(sender_counts)]
        sns.set_palette(colors)
        plt.pie(sender_counts['count'], labels=sender_counts['index'], autopct='%1.1f%%', startangle=90,
                wedgeprops=dict(width=0.3))
        plt.title('Distribution of Emails by Sender')

        # Draw a circle to create the "donut" aspect
        centre_circle = plt.Circle((0, 0), 0.6, color='white', linewidth=0.5)
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        st.pyplot(fig)



        word_counts = [len(subject.split()) for subject in subject_list]

        # Assuming 'timestamps' contains the list of timestamps
        # Convert timestamps to months for better visualization
        months = [timestamp.strftime('%b %Y') for timestamp in timestamps]

        # Create a DataFrame for the heatmap
        data = {'Months': months, 'Word Counts': word_counts}
        df = pd.DataFrame(data)

        # Pivot the DataFrame for heatmap
        heatmap_data = df.pivot_table(index='Months', columns='Word Counts', aggfunc='size', fill_value=0)

        # Plot the heatmap
        plt.figure(figsize=(5, 5))
        sns.heatmap(heatmap_data, cmap='viridis', cbar_kws={'label': 'Email Count'})
        plt.title('Spectrogram-like Analysis of Email Subjects',fontsize=7)
        st.set_option('deprecation.showPyplotGlobalUse', False)
        st.pyplot()


        # Show the plot

        # Create a Seaborn count plot
        months = [timestamp.strftime('%Y-%m') for timestamp in timestamps]

        # Create a Seaborn count plot
        plt.figure(figsize=(12, 6))
        sns.countplot(x=months, palette='viridis')
        plt.xticks(rotation=45)
        plt.xlabel('Month')
        plt.ylabel('Number of Emails')
        plt.title('Monthly Distribution of Emails')
        plt.tight_layout()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Display the count plot in Streamlit
        st.pyplot()

        x_values = [1, 2, 3, 4, 5]
        y_line = [10, 12, 16, 8, 15]
        y_bar = [5, 8, 12, 6, 10]

        # Seaborn line plot
        sns.lineplot(x=x_values, y=y_line, label='Line Plot', color='orange')

        # Matplotlib bar plot
        plt.bar(x=x_values, height=y_bar, alpha=0.5, label='Bar Plot', color='blue')

        # Set labels and title
        plt.xlabel('Categories')
        plt.ylabel('Emails Count')
        plt.title('Combo Chart')

        # Show legend
        plt.legend()
        st.set_option('deprecation.showPyplotGlobalUse', False)
        # Display the count plot in Streamlit
        st.pyplot()




if __name__=="__main__":
    main()




