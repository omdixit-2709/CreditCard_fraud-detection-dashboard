import React, { useState } from "react";
import { Button, Container, Form, Alert } from "react-bootstrap";
import axios from "axios";

function App() {
  const [image, setImage] = useState(null);
  const [username, setUsername] = useState("");
  const [message, setMessage] = useState("");
  const [isLogin, setIsLogin] = useState(false);  // Track if it's login or registration

  const handleImageChange = (e) => {
    setImage(e.target.files[0]);
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    const formData = new FormData();
    formData.append("image", image);
    formData.append("username", username);

    try {
      const endpoint = isLogin ? "/login" : "/register";
      const response = await axios.post(`http://127.0.0.1:5000${endpoint}`, formData);
      setMessage(response.data.message);
    } catch (error) {
      setMessage(error.response ? error.response.data.message : "Error occurred");
    }
  };

  return (
    <Container className="my-5">
      <h2>Facial Authentication</h2>
      <Form onSubmit={handleSubmit}>
        <Form.Group controlId="formUsername">
          <Form.Label>Username</Form.Label>
          <Form.Control
            type="text"
            placeholder="Enter username"
            value={username}
            onChange={(e) => setUsername(e.target.value)}
            required
          />
        </Form.Group>
        <Form.Group controlId="formFile" className="mt-3">
          <Form.Label>Upload Face Image</Form.Label>
          <Form.Control
            type="file"
            onChange={handleImageChange}
            accept="image/*"
            required
          />
        </Form.Group>
        <Button variant="primary" type="submit" className="mt-3">
          {isLogin ? "Login" : "Register"}
        </Button>
      </Form>

      <Button
        variant="secondary"
        onClick={() => setIsLogin(!isLogin)}
        className="mt-3"
      >
        Switch to {isLogin ? "Register" : "Login"}
      </Button>

      {message && <Alert className="mt-3" variant="info">{message}</Alert>}
    </Container>
  );
}

export default App