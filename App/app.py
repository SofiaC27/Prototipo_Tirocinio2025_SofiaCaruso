import streamlit as st
import sys
import os

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.db_manager import insert_data, read_data

# Function to save uploaded images into the Images folder
def save_image_to_folder(uploaded_file):
    folder_path = "Images"
    # Create the folder if it does not exist
    os.makedirs(folder_path, exist_ok=True)
    # Construct the full file path
    file_path = os.path.join(folder_path, uploaded_file.name)
    # Save the file in binary format
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path  # Return the saved file path

# Application title
st.title("Document Management")

# File upload functionality
uploaded_file = st.file_uploader("Upload a file (JPG, JPEG, PNG, PDF)", type=["jpg", "jpeg", "png", "pdf"])

# Display the uploaded file
if uploaded_file is not None:
    st.write("You uploaded:", uploaded_file.name)
    # Save the file to the Images folder and database
    if st.button("Save to database"):
        file_path = save_image_to_folder(uploaded_file)  # Save the file in the Images folder
        insert_data(file_path)  # Insert the file path into the database
        st.success(f"File successfully saved to '{file_path}' and the database!")
else:
    st.warning("Please upload a file to proceed.")

# Show saved data from the database
st.subheader("Saved data in the database:")
data = read_data()  # Retrieves data from the database
if data:
    for row in data:
        st.write(f"ID: {row[0]} | File Path: {row[1]} | Upload Date: {row[2]}")
else:
    st.info("No data available in the database.")

# Button to start the processing
if st.button("Process"):
    st.write("Processing the file...")  # Replace with OCR or other logic

