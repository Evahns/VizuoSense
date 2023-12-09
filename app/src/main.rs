use gtk::prelude::*;
use gtk::{Label, Window, WindowType, Button, Box, Orientation};

fn main() {
    // Initialize GTK
    gtk::init().expect("Failed to initialize GTK.");

    // Create the main window
    let window: Window = Window::new(WindowType::Toplevel);
    window.set_title("VizuoSense");
    window.set_default_size(640, 420);

    // Create a vertical box to hold the UI elements
    vbox.set_orientation(Orientation::Vertical);
    vbox.set_spacing(5);
    // Create buttons for Perception and OCR
    let button_perception = Button::with_label("Perception");
    let button_ocr = Button::with_label("OCR");

    // Connect signals for Perception and OCR buttons
    let label = Label::new(None);
    let label_clone = label.clone();
    button_perception.connect_clicked(move |_| {
        label_clone.set_text("Switching to Perception functionality");
    });

    let label_clone = label.clone();
    button_ocr.connect_clicked(move |_| {
        label_clone.set_text("Switching to OCR functionality");
    });

    // Pack buttons into the vertical box
    vbox.pack_start(&button_perception, false, false, 5);
    vbox.pack_start(&button_ocr, false, false, 5);
    vbox.pack_start(&label, false, false, 5);

    // Add the vbox to the window
    window.add(&vbox);

    // Connect the "destroy" signal to exit the GTK main loop
    window.connect_destroy(|_| {
        gtk::main_quit();
    });

    // Show all UI elements
    window.show_all();

    // Run the GTK main loop
    gtk::main();
}