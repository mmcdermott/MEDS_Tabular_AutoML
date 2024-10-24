function toggleFolder(folderId, event) {
  if (!folderId) {
    console.error('Folder ID is required');
    return;
  }

  const content = document.getElementById(folderId);
  if (!content) {
    console.error(`Element with ID ${folderId} not found`);
    return;
  }

  const folderItem = content.previousElementSibling;
  if (!folderItem) {
    console.error(`No folder item found for content ${folderId}`);
    return;
  }

  const isExpanded = folderItem.classList.contains('active');

  // Update ARIA attributes
  folderItem.setAttribute('role', 'button');
  folderItem.setAttribute('aria-expanded', !isExpanded);
  folderItem.setAttribute('aria-controls', folderId);
  content.setAttribute('role', 'region');

  // Toggle active state on folder item
  folderItem.classList.toggle('active');

  // Toggle visibility of content
  content.classList.toggle('visible');

  if (event) {
    event.stopPropagation();
  }
}

// Add keyboard support
document.addEventListener('keydown', (event) => {
  if (event.target.hasAttribute('aria-controls') &&
      (event.key === 'Enter' || event.key === ' ')) {
    event.preventDefault();
    toggleFolder(event.target.getAttribute('aria-controls'), event);
  }
});
