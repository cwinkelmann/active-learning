import streamlit as st
from glob import glob
from PIL import Image
import io
import pandas as pd
from streamlit_image_annotation import pointdet


def main():
    st.title("Image Editor with Point Annotation")

    # Define your label list
    label_list = ['deer', 'human', 'dog', 'penguin', 'flamingo', 'teddy bear']

    # Upload image
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    # Instruction panel
    with st.expander("Instructions for using the annotation tool", expanded=True):
        st.markdown("""
        ### How to use the annotation tool:
        - **Add points**: Click directly on the image
        - **Select label**: After adding a point, select a label from the dropdown
        - **Edit point**: Click on an existing point to select it, then drag to reposition
        - **Delete point**: Click on a point to select it, then press the Delete key or use the "Delete Selected Point" button
        - **Space key**: Toggle between label selection and point addition mode
        """)

    if uploaded_file is not None:
        # Create a temp file path for the uploaded image
        image_path = f"temp_image_{uploaded_file.name}"

        # Save the uploaded file temporarily
        with open(image_path, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Initialize result dictionary in session state if not exists
        if 'result_dict' not in st.session_state:
            st.session_state['result_dict'] = {}

        # Initialize this image in the result dictionary if not exists
        if image_path not in st.session_state['result_dict']:
            st.session_state['result_dict'][image_path] = {
                'points': [],  # Start with no points
                'labels': []  # Start with no labels
            }

        # Track selected point for deletion
        if 'selected_point_index' not in st.session_state:
            st.session_state['selected_point_index'] = None

        # Controls for manual deletion
        col1, col2 = st.columns(2)

        with col1:
            # Manual selection of point to delete (useful if keyboard delete doesn't work)
            if st.session_state['result_dict'][image_path]['points']:
                point_options = [f"Point {i + 1}: ({p[0]}, {p[1]}) - {label_list[l]}"
                                 for i, (p, l) in enumerate(zip(
                        st.session_state['result_dict'][image_path]['points'],
                        st.session_state['result_dict'][image_path]['labels']
                    )) if i < len(st.session_state['result_dict'][image_path]['labels'])]

                if point_options:
                    selected_point = st.selectbox(
                        "Select a point to highlight/delete:",
                        options=["None"] + point_options
                    )

                    if selected_point != "None":
                        st.session_state['selected_point_index'] = point_options.index(selected_point)

        with col2:
            # Button to delete the selected point
            if st.button("Delete Selected Point") and st.session_state['selected_point_index'] is not None:
                idx = st.session_state['selected_point_index']
                if idx < len(st.session_state['result_dict'][image_path]['points']):
                    st.session_state['result_dict'][image_path]['points'].pop(idx)

                    # Make sure we don't go out of bounds for labels
                    if idx < len(st.session_state['result_dict'][image_path]['labels']):
                        st.session_state['result_dict'][image_path]['labels'].pop(idx)

                    st.session_state['selected_point_index'] = None
                    st.experimental_rerun()

        # Add a button to clear all points
        if st.button("Clear All Points"):
            st.session_state['result_dict'][image_path]['points'] = []
            st.session_state['result_dict'][image_path]['labels'] = []
            st.experimental_rerun()

        # Use pointdet to annotate the image
        new_labels = pointdet(
            image_path=image_path,
            label_list=label_list,
            points=st.session_state['result_dict'][image_path]['points'],
            labels=st.session_state['result_dict'][image_path]['labels'],
            use_space=True,
            key=image_path
        )

        # Update points and labels if changes were made
        if new_labels is not None:
            st.session_state['result_dict'][image_path]['points'] = [v['point'] for v in new_labels]
            st.session_state['result_dict'][image_path]['labels'] = [v['label_id'] for v in new_labels]

        # Display the results
        st.subheader("Annotation Results")

        # Show points in a more readable table format
        if st.session_state['result_dict'][image_path]['points']:
            point_data = []
            for i, (point, label_id) in enumerate(zip(
                    st.session_state['result_dict'][image_path]['points'],
                    st.session_state['result_dict'][image_path]['labels']
            )):
                if i < len(st.session_state['result_dict'][image_path]['labels']):
                    point_data.append({
                        "Point #": i + 1,
                        "X": point[0],
                        "Y": point[1],
                        "Label": label_list[label_id]
                    })

            st.table(pd.DataFrame(point_data))
        else:
            st.info("No points added yet. Click on the image to add points.")

        # Create a download option for the annotations
        if st.session_state['result_dict'][image_path]['points']:
            # Convert the annotation data to a dataframe
            data = []
            for i in range(len(st.session_state['result_dict'][image_path]['points'])):
                if i < len(st.session_state['result_dict'][image_path]['labels']):
                    data.append({
                        'image': image_path,
                        'x': st.session_state['result_dict'][image_path]['points'][i][0],
                        'y': st.session_state['result_dict'][image_path]['points'][i][1],
                        'label_id': st.session_state['result_dict'][image_path]['labels'][i],
                        'label_name': label_list[st.session_state['result_dict'][image_path]['labels'][i]]
                    })

            df = pd.DataFrame(data)
            csv = df.to_csv(index=False).encode('utf-8')

            st.download_button(
                label="Download Annotations as CSV",
                data=csv,
                file_name="annotations.csv",
                mime="text/csv",
            )


if __name__ == "__main__":
    main()