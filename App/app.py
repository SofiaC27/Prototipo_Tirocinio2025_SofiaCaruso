import streamlit as st
import sys
import os
import time
import pandas as pd
import csv

# Add the project's root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from Database.db_manager import insert_data, read_data, delete_data

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
st.markdown("<h1 style='text-align: center; color: blue; font-size: 60px;'>Smart Receipts</h1>", unsafe_allow_html=True)
st.markdown("<h2 style='text-align: center; color: black; font-size: 25px;'>"
            "An advanced web application for uploading receipts and PDFs, extracting data with "
            "OCR, and organizing it in a searchable database. Enhanced with AI/LLM for natural"
            " language interaction and advanced analysis</h2>", unsafe_allow_html=True)


st.divider()
st.subheader("File Uploader")

# File upload functionality for multiple files
uploaded_files = st.file_uploader("Upload files (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)

# Check if files have been uploaded
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Display the name of the uploaded file
        st.write(f"File uploaded: {uploaded_file.name}")

        # Display a preview of the uploaded image
        if uploaded_file.type.startswith("image"):
            st.image(uploaded_file, caption=f"Preview of {uploaded_file.name}", use_container_width=True)

        # Button to save the file to the folder and database
        if st.button(f"Save {uploaded_file.name} to database"):
            file_path = save_image_to_folder(uploaded_file)  # Function to save the file to the folder Images
            insert_data(file_path)  # Function to insert the file path into the database
            st.success(f"File successfully saved to '{file_path}' and the database!")

        # Simulated progress bar for processing the file
        with st.spinner("Processing..."):
            progress = st.progress(0)
            for i in range(100):  # Simulate file processing
                time.sleep(0.01)
                progress.progress(i + 1)
            st.success(f"{uploaded_file.name} processed successfully!")
else:
    # Warning message if no files are uploaded
    st.warning("Please upload a file to proceed.")

st.divider()
st.subheader("Database Management")

# Show saved data from the database
st.subheader("Saved data in the database:")
data = read_data()  # Retrieves data from the database

# Check if there is data in the database
if data:
    # Enhanced visualization - Table
    df = pd.DataFrame(data, columns=["ID", "File Path", "Upload Date"])
    st.dataframe(df)  # Display the data in a table format for better readability

    # Option to delete files
    file_to_delete = st.selectbox("Select file to delete", [row[1] for row in data])  # Dropdown to select a file for deletion
    if st.button("Delete file"):  # Button to confirm deletion
        delete_data(file_to_delete)  # Replace with the actual function to delete the file from the database
        st.success(f"File '{file_to_delete}' successfully deleted!")  # Success message after deletion

    # Pagination
    items_per_page = 10
    total_pages = (len(data) + items_per_page - 1) // items_per_page  # Calculate the total number of pages

    if total_pages > 1:  # Only display the slider if more than one page exists
        page = st.slider("Page", 1, total_pages)  # Slider for page selection
        start = (page - 1) * items_per_page
        end = start + items_per_page
        for row in data[start:end]:
            st.write(f"ID: {row[0]} | File Path: {row[1]} | Upload Date: {row[2]}")
    else:
        # Display all data if only one page exists
        for row in data:
            st.write(f"ID: {row[0]} | File Path: {row[1]} | Upload Date: {row[2]}")

    # Export data to CSV
    if st.button("Export data to CSV"):  # Button to export data
        with open("exported_data.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["ID", "File Path", "Upload Date"])  # Write column headers to the CSV
            writer.writerows(data)  # Write rows of data to the CSV
        st.success("Data successfully exported to 'exported_data.csv'!")  # Success message after export
else:
    # Inform the user if no data is available
    st.info("No data available in the database.")  # Informational message when the database is empty



# Button to start the processing
if st.button("Process"):
    st.write("Processing the file...")  # Replace with OCR or other logic


